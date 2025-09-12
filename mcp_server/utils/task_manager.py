from utils.tasks import Task

class TaskManager:
    def __init__(self):
        self.task_list: list[Task] = [] 
        self._current_id = 0
        
    def get_tasks(self) -> list[Task]:
        return self.task_list
    
    def add_task(self, task_name: str, is_completed: bool = False) -> None:
        task = Task(id=self._generate_id(), name=task_name, is_completed=is_completed)
        self.task_list.append(task)
        
    def remove_task(self, task_id: str) -> None:
        self.task_list = [task for task in self.task_list if task.id != task_id]
        
    def mark_complete(self, task_id: str) -> None:
        self.task_list = [
            task if task.id != task_id else Task(id=task.id, name=task.name, is_completed=True)
            for task in self.task_list 
        ]
        
    def _generate_id(self) -> str:
        self._current_id += 1
        return str(self._current_id)