import os
import re
import smtplib
import threading
from datetime import date, datetime
from email.mime.text import MIMEText
from typing import Annotated, Literal, Optional, TypedDict

import requests
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


APP_TITLE = "Developer Productivity Agent"
DEFAULT_MODEL = "openai/gpt-oss-120b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_TIMEOUT_SECONDS = 45
ESTIMATE_TIMEOUT_SECONDS = 25


def send_reminder_email(
    to_email: str,
    gmail_user: str,
    gmail_password: str,
    subject: str,
    body: str,
) -> None:
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, to_email, msg.as_string())


def collect_due_tasks(tasks: dict) -> list[dict]:
    today = date.today().isoformat()
    relevant = []
    for task in tasks.values():
        if task["status"] != "pending" or not task["deadline"]:
            continue
        if task["deadline"] > today:
            continue
        if task.get("last_reminder_sent") == today:
            continue
        relevant.append(task)
    return relevant


def run_reminder_check(runtime: dict) -> str:
    settings = runtime["settings"]
    tasks = runtime["tasks_ref"][0]
    if tasks is None:
        return "No tasks loaded yet."
    if not settings["enabled"]:
        return "Auto reminders disabled."
    if not settings["recipient"] or not settings["gmail_user"] or not settings["gmail_password"]:
        return "Missing recipient or Gmail credentials."

    due_tasks = collect_due_tasks(tasks)
    if not due_tasks:
        runtime["last_check"][0] = datetime.now().isoformat(timespec="seconds")
        return "No deadlines today."

    today = date.today().isoformat()
    task_list = "\n".join(
        f"- [{task['id']}] {task['title']} (due {task['deadline']})" for task in due_tasks
    )
    subject = f"Deadline reminder - {len(due_tasks)} task(s) due"
    body = f"Hi,\n\nThese tasks are due today or overdue:\n{task_list}\n\nGood luck!\n"

    try:
        send_reminder_email(
            settings["recipient"],
            settings["gmail_user"],
            settings["gmail_password"],
            subject,
            body,
        )
    except Exception as exc:
        return f"Email error: {exc}"

    for task in due_tasks:
        task["last_reminder_sent"] = today
    runtime["last_check"][0] = datetime.now().isoformat(timespec="seconds")
    return f"Sent reminder for {len(due_tasks)} task(s)."


@st.cache_resource(show_spinner=False)
def get_reminder_runtime() -> dict:
    runtime = {
        "settings": {
            "enabled": False,
            "recipient": "",
            "gmail_user": "",
            "gmail_password": "",
        },
        "tasks_ref": [None],
        "stop_event": threading.Event(),
        "last_check": [None],
        "started": [False],
    }

    def loop() -> None:
        stop_event: threading.Event = runtime["stop_event"]
        while not stop_event.wait(60 * 30):
            try:
                run_reminder_check(runtime)
            except Exception:
                pass

    threading.Thread(target=loop, daemon=True).start()
    runtime["started"][0] = True
    return runtime


def normalize_github_repo(repo_input: str) -> str:
    """Accept owner/repo or a GitHub URL and return owner/repo."""
    value = repo_input.strip().removesuffix("/")
    if not value:
        return ""

    match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/#?]+)", value)
    if match:
        return f"{match.group('owner')}/{match.group('repo').removesuffix('.git')}"

    return value.removesuffix(".git")


class TaskEstimate(BaseModel):
    task_title: str = Field(description="Title of the task")
    estimated_hours: float = Field(description="Estimated hours to complete")
    complexity: Literal["low", "medium", "high"] = Field(description="Task complexity")
    reasoning: str = Field(description="Why this estimate was given")
    suggested_breakdown: list[str] = Field(description="Step-by-step subtasks")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM_PROMPT = """You are an intelligent Developer Productivity Agent.

Your job is to help developers stay organized and focused. You can:
- Manage tasks with deadlines and estimates
- Analyze GitHub repos to track issues and recent activity
- Research blockers and estimate task effort using web search
- Send deadline reminder emails after human confirmation
- Generate daily briefings

- When adding a task, estimate missing time automatically before storing it.
- If the user selected a GitHub repo in the UI, treat it as the default repo context.
- When calling add_task, pass the selected GitHub repo as the repo argument unless the user gives a different repo.
- When the user asks about "the repo" or "this repo", use the selected GitHub repo.
- Treat natural sentences like "I need to implement a login page" as requests to add a task.
- Infer a short title and useful description when the user does not use a strict command format.
- When the user asks to set/change a deadline, status, estimate, title, or description, call update_task.
- Never call delete_task unless the user clearly asks to delete/remove a task.
Be concise, practical, and developer-friendly.
"""


@st.cache_resource(show_spinner=False)
def build_agent_runtime(api_key: str, model: str):
    os.environ["OPENAI_API_KEY"] = api_key

    tasks: dict[str, dict] = {}
    counter = {"value": 0}
    github_headers = {"Accept": "application/vnd.github+json"}
    ddg = DuckDuckGoSearchRun()

    def next_id() -> str:
        counter["value"] += 1
        return f"task_{counter['value']:03d}"

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=2048,
        timeout=CHAT_TIMEOUT_SECONDS,
        max_retries=1,
    )

    estimator_llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_tokens=1500,
        timeout=ESTIMATE_TIMEOUT_SECONDS,
        max_retries=0,
    ).with_structured_output(TaskEstimate)

    def quick_estimate_hours(title: str, description: str, repo_context: str) -> Optional[float]:
        """Ask the plain LLM for a single hour number when structured output fails."""
        try:
            response = llm.invoke(
                "You are a senior developer. Estimate how many hours this task will take "
                "a mid-level developer. Reply with ONLY a number, no units or words.\n\n"
                f"Task: {title}\nDescription: {description}{repo_context}"
            )
            text = str(response.content).strip().split()[0]
            return float(text.replace("h", "").replace(",", "."))
        except Exception:
            return None

    def estimate_task_details(
        title: str,
        description: str,
        repo: Optional[str] = None,
        use_web: bool = True,
    ) -> tuple[Optional[TaskEstimate], str]:
        web_context = ""
        if use_web:
            try:
                web_context = ddg.run(f"how long does it take to {title} programming")[:1000]
            except Exception:
                web_context = "No web context available."

        repo_context = ""
        if repo:
            try:
                owner, repo_name = repo.split("/", 1)
                response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo_name}",
                    headers=github_headers,
                    timeout=10,
                )
                data = response.json()
                repo_context = (
                    f"\nRepo context: {data.get('language', 'unknown')} project, "
                    f"{data.get('size', 0)} KB, {data.get('open_issues_count', 0)} open issues."
                )
            except Exception:
                pass

        prompt = (
            f"Task: {title}\n"
            f"Description: {description}{repo_context}\n\n"
            f"Web research context:\n{web_context}\n\n"
            f"Provide a realistic time estimate for a mid-level developer."
        )

        try:
            return estimator_llm.invoke(prompt), ""
        except Exception as exc:
            hours = quick_estimate_hours(title, description, repo_context)
            if hours is not None:
                return (
                    TaskEstimate(
                        task_title=title,
                        estimated_hours=hours,
                        complexity="medium",
                        reasoning="Quick estimate (structured output unavailable).",
                        suggested_breakdown=[],
                    ),
                    "",
                )
            return None, f"Automatic estimate skipped because the estimator failed: {exc}"

    @tool
    def add_task(
        title: str,
        description: str,
        deadline: Optional[str] = None,
        repo: Optional[str] = None,
        estimated_hours: Optional[float] = None,
    ) -> str:
        """Add a new task to the task tracker.

        If estimated_hours is missing, estimate it before saving the task.
        """
        estimate_text = ""
        if estimated_hours is None:
            estimate, fallback = estimate_task_details(title, description, repo, use_web=False)
            if estimate:
                estimated_hours = estimate.estimated_hours
                estimate_text = (
                    f"\nEstimated time: {estimate.estimated_hours}h\n"
                    f"Complexity: {estimate.complexity}\n"
                    f"Reasoning: {estimate.reasoning}\n"
                    f"Breakdown:\n"
                    + "\n".join(f"- {step}" for step in estimate.suggested_breakdown)
                )
            elif fallback:
                estimate_text = f"\nEstimate note:\n{fallback}"

        task_id = next_id()
        tasks[task_id] = {
            "id": task_id,
            "title": title,
            "description": description,
            "deadline": deadline,
            "repo": repo,
            "estimated_hours": estimated_hours,
            "actual_hours": None,
            "status": "pending",
            "created_at": date.today().isoformat(),
        }
        due = f" (due {deadline})" if deadline else ""
        estimate_suffix = f" | est: {estimated_hours}h" if estimated_hours else ""
        return f"Task added: [{task_id}] {title}{due}{estimate_suffix}{estimate_text}"

    @tool
    def list_tasks(status_filter: Optional[str] = None) -> str:
        """List all tasks, optionally filtered by status: pending or done."""
        filtered = list(tasks.values())
        if status_filter:
            filtered = [t for t in filtered if t["status"] == status_filter]
        if not filtered:
            return "No tasks found."

        lines = []
        for task in filtered:
            deadline = f" | due: {task['deadline']}" if task["deadline"] else ""
            repo = f" | repo: {task['repo']}" if task["repo"] else ""
            estimate = f" | est: {task['estimated_hours']}h" if task["estimated_hours"] else ""
            lines.append(f"[{task['id']}] [{task['status'].upper()}] {task['title']}{deadline}{repo}{estimate}")
        return "\n".join(lines)

    @tool
    def complete_task(task_id: str, actual_hours: Optional[float] = None) -> str:
        """Mark a task as done and optionally record actual hours."""
        if task_id not in tasks:
            return f"Task {task_id} not found."
        tasks[task_id]["status"] = "done"
        tasks[task_id]["actual_hours"] = actual_hours
        tasks[task_id]["completed_at"] = date.today().isoformat()
        return f"Marked {task_id} ('{tasks[task_id]['title']}') as done."

    @tool
    def update_task(
        task_id: Optional[str] = None,
        title_contains: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        deadline: Optional[str] = None,
        repo: Optional[str] = None,
        estimated_hours: Optional[float] = None,
        status: Optional[Literal["pending", "done"]] = None,
    ) -> str:
        """Update an existing task.

        Args:
            task_id: Exact task ID, such as task_001. Use this when available.
            title_contains: Part of the task title to search for when task_id is unknown.
            title: New task title. Optional.
            description: New task description. Optional.
            deadline: New due date in YYYY-MM-DD format. Optional.
            repo: GitHub repo in owner/repo format. Optional.
            estimated_hours: New estimated hours. Optional.
            status: New status, either pending or done. Optional.
        """
        matched_task = None
        if task_id:
            matched_task = tasks.get(task_id)
        elif title_contains:
            search_text = title_contains.lower()
            matches = [task for task in tasks.values() if search_text in task["title"].lower()]
            if len(matches) > 1:
                return "Multiple tasks matched. Please specify the task ID:\n" + "\n".join(
                    f"- [{task['id']}] {task['title']}" for task in matches
                )
            if matches:
                matched_task = matches[0]

        if not matched_task:
            return "Task not found. Ask to list tasks, then update using the task ID."

        if title is not None:
            matched_task["title"] = title
        if description is not None:
            matched_task["description"] = description
        if deadline is not None:
            matched_task["deadline"] = deadline
        if repo is not None:
            matched_task["repo"] = repo
        if estimated_hours is not None:
            matched_task["estimated_hours"] = estimated_hours
        if status is not None:
            matched_task["status"] = status
            if status == "done":
                matched_task["completed_at"] = date.today().isoformat()

        deadline_text = f" | due: {matched_task['deadline']}" if matched_task["deadline"] else ""
        repo_text = f" | repo: {matched_task['repo']}" if matched_task["repo"] else ""
        estimate_text = f" | est: {matched_task['estimated_hours']}h" if matched_task["estimated_hours"] else ""
        return (
            f"Updated [{matched_task['id']}] {matched_task['title']}"
            f"{deadline_text}{repo_text}{estimate_text}"
        )

    @tool
    def delete_task(task_id: str) -> str:
        """Permanently delete a task. Human approval is required."""
        if task_id not in tasks:
            return f"Task {task_id} not found."
        task_title = tasks[task_id]["title"]
        approval = interrupt(
            f"DELETE TASK\n\nID: {task_id}\nTitle: {task_title}\n\nType yes to confirm, or no to cancel."
        )
        if str(approval).strip().lower() == "yes":
            del tasks[task_id]
            return f"Task {task_id} ('{task_title}') deleted."
        return f"Deletion of {task_id} cancelled."

    @tool
    def check_deadlines() -> str:
        """Check which tasks are due today or overdue."""
        today = date.today().isoformat()
        due_today, overdue = [], []
        for task in tasks.values():
            if task["status"] == "done" or not task["deadline"]:
                continue
            if task["deadline"] == today:
                due_today.append(task)
            elif task["deadline"] < today:
                overdue.append(task)

        lines = []
        if due_today:
            lines.append("Due today:")
            lines.extend(f"- [{task['id']}] {task['title']}" for task in due_today)
        if overdue:
            lines.append("Overdue:")
            lines.extend(f"- [{task['id']}] {task['title']} (was due {task['deadline']})" for task in overdue)
        return "\n".join(lines) if lines else "No tasks due today or overdue."

    @tool
    def get_github_repo_info(owner: str, repo: str) -> str:
        """Fetch general info about a public GitHub repository."""
        response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=github_headers,
            timeout=10,
        )
        if response.status_code == 404:
            return f"Repository {owner}/{repo} not found."
        data = response.json()
        return (
            f"Repo: {data['full_name']}\n"
            f"Description: {data.get('description', 'N/A')}\n"
            f"Language: {data.get('language', 'N/A')}\n"
            f"Stars: {data.get('stargazers_count', 0):,}\n"
            f"Open issues: {data.get('open_issues_count', 0)}\n"
            f"Last pushed: {data.get('pushed_at', 'N/A')[:10]}"
        )

    @tool
    def get_github_issues(owner: str, repo: str, max_results: int = 5) -> str:
        """Fetch recent open issues from a public GitHub repository."""
        response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/issues",
            headers=github_headers,
            params={"state": "open", "per_page": min(max_results, 10)},
            timeout=10,
        )
        issues = response.json()
        if not issues or isinstance(issues, dict):
            return f"No open issues found for {owner}/{repo}."
        lines = [f"Open issues in {owner}/{repo}:"]
        for issue in issues:
            lines.append(f"#{issue['number']}: {issue['title']} ({issue.get('comments', 0)} comments)")
        return "\n".join(lines)

    @tool
    def get_github_recent_commits(owner: str, repo: str, max_results: int = 5) -> str:
        """Fetch recent commits from a public GitHub repository."""
        response = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            headers=github_headers,
            params={"per_page": min(max_results, 10)},
            timeout=10,
        )
        commits = response.json()
        if not commits or isinstance(commits, dict):
            return f"No commits found for {owner}/{repo}."
        lines = [f"Recent commits in {owner}/{repo}:"]
        for commit in commits:
            msg = commit["commit"]["message"].split("\n")[0][:80]
            author = commit["commit"]["author"]["name"]
            date_str = commit["commit"]["author"]["date"][:10]
            lines.append(f"[{date_str}] {author}: {msg}")
        return "\n".join(lines)

    @tool
    def search_web(query: str) -> str:
        """Search the web using DuckDuckGo."""
        return ddg.run(query)[:2000]

    @tool
    def estimate_task(title: str, description: str, repo: Optional[str] = None) -> str:
        """Estimate task complexity, hours, and suggested breakdown."""
        estimate, fallback = estimate_task_details(title, description, repo, use_web=True)
        if not estimate:
            return f"Task Estimate (plain):\n{fallback}"
        return (
            f"Task Estimate: {estimate.task_title}\n"
            f"Estimated time: {estimate.estimated_hours}h\n"
            f"Complexity: {estimate.complexity}\n"
            f"Reasoning: {estimate.reasoning}\n"
            "Breakdown:\n"
            + "\n".join(f"- {step}" for step in estimate.suggested_breakdown)
        )

    @tool
    def send_deadline_email(to_email: str, task_ids: str) -> str:
        """Send a deadline reminder email. Human approval is required."""
        ids = [task_id.strip() for task_id in task_ids.split(",")]
        tasks_to_notify = [tasks[task_id] for task_id in ids if task_id in tasks]
        if not tasks_to_notify:
            return "No valid tasks found for those IDs."

        task_list = "\n".join(
            f"- {task['title']} (due: {task['deadline'] or 'no deadline'})" for task in tasks_to_notify
        )
        subject = f"Deadline Reminder - {len(tasks_to_notify)} task(s) due"
        body = f"Hi,\n\nYou have these task(s) due:\n{task_list}\n\nGood luck!\n"

        approval = interrupt(
            f"SEND EMAIL\n\nTo: {to_email}\nSubject: {subject}\n\n{task_list}\n\nType yes to send, or no to cancel."
        )
        if str(approval).strip().lower() != "yes":
            return "Email cancelled."

        gmail_user = os.environ.get("GMAIL_USER", "")
        gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD", "")
        if gmail_user and gmail_app_password:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = gmail_user
            msg["To"] = to_email
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(gmail_user, gmail_app_password)
                server.sendmail(gmail_user, to_email, msg.as_string())
            return f"Email sent to {to_email}."

        return (
            f"[DRY RUN] Email would be sent to {to_email}:\n"
            f"Subject: {subject}\n\n{body}\n"
            "Set GMAIL_USER and GMAIL_APP_PASSWORD to send real emails."
        )

    @tool
    def daily_briefing() -> str:
        """Generate a daily briefing with deadlines and pending tasks."""
        today = date.today().isoformat()
        pending = [task for task in tasks.values() if task["status"] == "pending"]
        done_today = [task for task in tasks.values() if task["status"] == "done" and task.get("completed_at") == today]
        due_today = [task for task in pending if task["deadline"] == today]
        overdue = [task for task in pending if task["deadline"] and task["deadline"] < today]
        total_estimated = sum(task["estimated_hours"] or 0 for task in pending)

        lines = [
            f"Daily Briefing - {today}",
            f"Summary: {len(pending)} pending task(s), {len(done_today)} completed today",
            f"Total estimated work: {total_estimated:.1f}h",
        ]
        if overdue:
            lines.append("\nOverdue:")
            lines.extend(f"- [{task['id']}] {task['title']} (was due {task['deadline']})" for task in overdue)
        if due_today:
            lines.append("\nDue today:")
            lines.extend(f"- [{task['id']}] {task['title']}" for task in due_today)
        if pending:
            lines.append("\nPending tasks:")
            for task in sorted(pending, key=lambda item: item["deadline"] or "9999"):
                deadline = f" | due: {task['deadline']}" if task["deadline"] else ""
                estimate = f" | est: {task['estimated_hours']}h" if task["estimated_hours"] else ""
                lines.append(f"- [{task['id']}] {task['title']}{deadline}{estimate}")
        else:
            lines.append("\nNo pending tasks.")
        return "\n".join(lines)

    all_tools = [
        add_task,
        list_tasks,
        complete_task,
        update_task,
        delete_task,
        check_deadlines,
        get_github_repo_info,
        get_github_issues,
        get_github_recent_commits,
        search_web,
        estimate_task,
        send_deadline_email,
        daily_briefing,
    ]
    llm_with_tools = llm.bind_tools(all_tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "__end__"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")

    graph = builder.compile(checkpointer=MemorySaver())
    return {"graph": graph, "tasks": tasks}


def get_interrupt_prompt(graph, thread_id: str) -> Optional[str]:
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        interrupts = getattr(task, "interrupts", None)
        if interrupts:
            return str(interrupts[0].value)
    return None


def run_graph(graph, thread_id: str, message: str, selected_repo: str = "") -> tuple[str, Optional[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    agent_message = message
    if selected_repo:
        agent_message = (
            f"Selected GitHub repo context: {selected_repo}\n"
            f"Use this as the default repo if the user says 'this repo' or does not provide another repo.\n"
            f"When adding a task, call add_task with repo='{selected_repo}' unless the user explicitly gives another repo.\n\n"
            f"User message: {message}"
        )
    result = graph.invoke({"messages": [HumanMessage(content=agent_message)]}, config=config)
    pending_prompt = get_interrupt_prompt(graph, thread_id)
    if pending_prompt:
        return "[Paused for approval]", pending_prompt
    return str(result["messages"][-1].content), None


def resume_graph(graph, thread_id: str, approval: str) -> tuple[str, Optional[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=approval), config=config)
    pending_prompt = get_interrupt_prompt(graph, thread_id)
    if pending_prompt:
        return "[Paused for approval]", pending_prompt
    return str(result["messages"][-1].content), None


st.set_page_config(page_title=APP_TITLE, page_icon="🤖")
st.title(APP_TITLE)
st.caption("LangGraph task manager, GitHub helper, web researcher, and deadline reminder chat.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenRouter API key",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Use your OpenRouter key. It is stored only for this Streamlit session.",
    )
    model = st.text_input("Model", value=DEFAULT_MODEL)
    thread_id = st.text_input("Thread ID", value=st.session_state.get("thread_id", "default"))
    st.session_state.thread_id = thread_id

    repo_input = st.text_input(
        "GitHub repo link",
        value=st.session_state.get("selected_repo", ""),
        placeholder="https://github.com/your-username/your-repo",
        help="Enter your repo first. You can paste a full GitHub URL or owner/repo.",
    ).strip()
    selected_repo = normalize_github_repo(repo_input)
    st.session_state.selected_repo = selected_repo
    if selected_repo:
        st.success(f"Tasks will be linked to `{selected_repo}`")

    st.divider()
    st.subheader("Gmail (for reminders)")
    gmail_user = st.text_input("GMAIL_USER", value=os.environ.get("GMAIL_USER", ""))
    gmail_password = st.text_input("GMAIL_APP_PASSWORD", type="password", value=os.environ.get("GMAIL_APP_PASSWORD", ""))
    if gmail_user:
        os.environ["GMAIL_USER"] = gmail_user
    if gmail_password:
        os.environ["GMAIL_APP_PASSWORD"] = gmail_password

    st.divider()
    st.subheader("Automatic deadline reminders")
    auto_reminders = st.checkbox(
        "Email me automatically when a task is due today or overdue",
        value=st.session_state.get("auto_reminders", False),
        help="While Streamlit is running, the app emails you once per day for each due task. No approval prompt.",
    )
    st.session_state.auto_reminders = auto_reminders
    reminder_recipient = st.text_input(
        "Reminder recipient",
        value=st.session_state.get("reminder_recipient", gmail_user),
        placeholder="you@example.com",
    ).strip()
    st.session_state.reminder_recipient = reminder_recipient
    send_now = st.button("Send reminder now", disabled=not (auto_reminders and reminder_recipient and gmail_user and gmail_password))

    if st.button("Reset app memory"):
        st.cache_resource.clear()
        st.session_state.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! Paste your GitHub repo link in the sidebar first. Then you can talk normally, like: \"I need to implement a login page with frontend validation.\"",
        }
    ]

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if not api_key:
    st.warning("Enter your OpenRouter API key in the sidebar to start chatting.")
    st.stop()

if not selected_repo:
    st.info("Paste your GitHub repo link in the sidebar first. After that, every new task will be linked to that repo.")
    st.stop()

runtime = build_agent_runtime(api_key, model)
graph = runtime["graph"]

reminder_runtime = get_reminder_runtime()
reminder_runtime["tasks_ref"][0] = runtime["tasks"]
reminder_runtime["settings"].update(
    {
        "enabled": auto_reminders,
        "recipient": reminder_recipient,
        "gmail_user": gmail_user,
        "gmail_password": gmail_password,
    }
)

if auto_reminders and reminder_recipient and gmail_user and gmail_password:
    status = run_reminder_check(reminder_runtime)
    if status.startswith("Sent reminder"):
        st.toast(status)

if send_now:
    status = run_reminder_check(reminder_runtime)
    st.toast(status)

with st.sidebar:
    last_check = reminder_runtime["last_check"][0]
    if last_check:
        st.caption(f"Last reminder check: {last_check}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_approval:
    st.warning(st.session_state.pending_approval)
    col1, col2 = st.columns(2)
    with col1:
        approve = st.button("Approve", type="primary")
    with col2:
        cancel = st.button("Cancel")

    if approve or cancel:
        approval = "yes" if approve else "no"
        st.session_state.messages.append({"role": "user", "content": approval})
        with st.spinner("Resuming agent..."):
            answer, pending = resume_graph(graph, thread_id, approval)
        st.session_state.pending_approval = pending
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

prompt = st.chat_input("Message the agent...", disabled=bool(st.session_state.pending_approval))
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, pending = run_graph(graph, thread_id, prompt, selected_repo)
                st.session_state.pending_approval = pending
            except Exception as exc:
                answer = f"Something went wrong: `{exc}`"
                st.session_state.pending_approval = None
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
