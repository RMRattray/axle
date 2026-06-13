import wikipediaapi
import os
from dotenv import load_dotenv

load_dotenv()
W = wikipediaapi.Wikipedia(user_agent=os.getenv("USER_AGENT"), language="en")

for category in ["People", "History", "Geography", "Arts", "Everyday_life", "Philosophy_and_religion", 
"Society_and_social_sciences", "Biology_and_health_sciences", "Physical_sciences", "Mathematics", "Technology"]:
    print(category)
    for title, page in W.page(f"Wikipedia:Vital_articles/Level_4/{category}").links.items():
        if ':' not in title and '/' not in title:
            with open(f"training_data/{title}.txt", "w") as file:
                file.write(page.text)