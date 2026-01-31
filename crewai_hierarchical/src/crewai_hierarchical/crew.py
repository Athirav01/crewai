from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


@CrewBase
class TestLlm:
    """TestLlm crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # -------------------- Agents --------------------

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            verbose=True
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"],
            verbose=True
        )

    @agent
    def educator(self) -> Agent:
        return Agent(
            config=self.agents_config["educator"],
            verbose=True
        )

    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["manager"],
            allow_delegation=True,
            verbose=True
        )

    # -------------------- Tasks --------------------

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"]
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],
            output_file="report.md",
            context=[self.analysis_task()]
        )

    # -------------------- Crew --------------------

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.researcher(),
                self.analyst(),
                self.educator(),
            ],
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.manager(),
            planning=False,
            max_iterations=1,
            verbose=False,
            allow_delegation=False  # 🔴 IMPORTANT
        )

