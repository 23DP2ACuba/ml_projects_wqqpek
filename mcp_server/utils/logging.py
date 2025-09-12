from datetime import datetime
def log_panel(title:str, content:str | None = None):
    with open("log.txt", "w+") as f:
        date = datetime.now()
        
        msg = {"date": date,"title": title, "content": content if content is not None else ""}
        
        f.write(msg)