from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dateutil import parser
import json
import os
from openai import OpenAI
import base64
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
import subprocess

app = FastAPI()

class TaskInput(BaseModel):
    task: str

client = OpenAI(api_key=os.environ["AIPROXY_TOKEN"])

data_files_generated = False

@app.post("/run")
def run_task(input: TaskInput):
    global data_files_generated
    try:
        # GPT-4o Task Classification
        prompt = f"""
        Map the following task description to one of these task labels (A1, A2, ..., A10):
        - A1: Download the required files.
        - A2: Format /data/format.md markdown file using prettier.
        - A3: Count the number of Wednesdays in /data/dates.txt.
        - A4: Sort contacts in /data/contacts.json.
        - A5: Extract first lines from 10 most recent log files.
        - A6: Extract H1 titles from markdown files in /data/docs/.
        - A7: Extract sender's email from /data/email.txt.
        - A8: Extract credit card number from /data/credit-card.png.
        - A9: Find most similar comments in /data/comments.txt.
        - A10: Calculate sales for 'Gold' tickets in /data/ticket-sales.db.

        User Task Description: {input.task}

        Respond with only the task label (e.g., A1, A2, ..., A10).
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Identify the task based on the user's description."},
                {"role": "user", "content": prompt}
            ]
        )

        task_label = response.choices[0].message.content.strip()

        if task_label == "A1" or not data_files_generated:
            user_email = "23f1000799@ds.study.iitm.ac.in"
            subprocess.run(["pip", "install", "uv"])
            subprocess.run(["curl", "-O", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"])
            subprocess.run(["python", "datagen.py", user_email])
            data_files_generated = True
            return {"status": "Data generation script executed"}

        elif task_label == "A2":
            try:
                subprocess.run(["npm", "install", "-g", "prettier@3.4.2"])
                subprocess.run(["prettier", "--write", "data/format.md"], check=True)
                return {"status": "Markdown file formatted successfully"}
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"Formatting failed: {str(e)}")
        elif task_label == "A3":
            try:
                with open("data/dates.txt", "r") as file:
                    dates = file.readlines()

                wednesday_count = 0

                for date in dates:
                    date = date.strip()
                    if not date:
                        continue
                    try:
                        parsed_date = parser.parse(date)
                        if parsed_date.weekday() == 2:
                            wednesday_count += 1
                    except Exception as e:
                        print(f"Skipping invalid date: {date} - Error: {e}")

                with open("data/dates-wednesdays.txt", "w") as file:
                    file.write(str(wednesday_count))

                return {"status": "Wednesdays counted", "wednesday_count": wednesday_count}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing dates.txt: {str(e)}")}

        elif task_label == "A4":
            try:
                with open("data/contacts.json", "r") as file:
                    contacts = json.load(file)

                sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

                with open("data/contacts-sorted.json", "w") as file:
                    json.dump(sorted_contacts, file, indent=4)

                return {"status": "Contacts sorted successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error sorting contacts: {str(e)}")

        elif task_label == "A5":
            try:
                log_files = [f for f in os.listdir("data/logs") if f.endswith(".log")]
                log_files.sort(key=lambda x: os.path.getmtime(os.path.join("data/logs", x)), reverse=True)

                recent_logs = log_files[:10]

                first_lines = []

                for log in recent_logs:
                    with open(os.path.join("data/logs", log), "r") as file:
                        first_line = file.readline().strip()
                        first_lines.append(first_line)

                with open("data/logs-recent.txt", "w") as output_file:
                    output_file.write("\n".join(first_lines))

                return {"status": "Recent log lines extracted successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing logs: {str(e)}")

        elif task_label == "A6":
            try:
                docs_path = "data/docs"
                index = {}

                for root, _, files in os.walk(docs_path):
                    for file in files:
                        if file.endswith(".md"):
                            with open(os.path.join(root, file), 'r') as f:
                                for line in f:
                                    if line.startswith("# "):
                                        index[file] = line[2:].strip()
                                        break

                with open("data/docs/index.json", "w") as index_file:
                    json.dump(index, index_file, indent=4)

                return {"status": "Docs index created successfully"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating docs index: {str(e)}")
        elif task_label == "A7":
            try:
                
                with open("data/email.txt", "r") as file:
                    email_content = file.read()

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract the sender's email address from the provided email content."},
                        {"role": "user", "content": email_content}
                            ],
                    max_tokens=50
                )

                sender_email = response.choices[0].message.content.strip()

                with open("data/email-sender.txt", "w") as output_file:
                    output_file.write(sender_email)

                return {"status": "Sender email extracted", "sender_email": sender_email}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error extracting email sender: {str(e)}")


        elif task_label == "A8":
            try:
                
                with open("data/credit-card.png", "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Extract the credit card number from the image as a continuous string of digits without spaces."},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Extract the credit card number from this image:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ],
                    max_tokens=50
                )

                card_number = response.choices[0].message.content.strip()

                with open("data/credit-card.txt", "w") as output_file:
                    output_file.write(card_number)

                return {"status": "Credit card number extracted", "card_number": card_number}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error extracting credit card number: {str(e)}")

        elif task_label == "A9":
            try:
                                
                def get_embedding(text):
                    response = client.embeddings.create(
                        input=text,
                        model="text-embedding-3-small"
                    )
                    return response.data[0].embedding

                with open("data/comments.txt", "r", encoding="utf-8") as file:
                    comments = [line.strip() for line in file.readlines() if line.strip()]

                if len(comments) < 2:
                    raise HTTPException(status_code=400, detail="Not enough comments to compare.")

                embeddings = [get_embedding(comment) for comment in comments]

                similarities = cosine_similarity(embeddings)
                most_similar = (-np.triu(similarities, k=1)).argmax()
                i, j = divmod(most_similar, len(comments))

                with open("data/comments-similar.txt", "w", encoding="utf-8") as file:
                    file.write(comments[i] + "\n")
                    file.write(comments[j] + "\n")

                return {
                    "status": "Most similar comments found",
                    "comment_1": comments[i],
                    "comment_2": comments[j]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error finding similar comments: {str(e)}")

        elif task_label == "A10":
            try:
                db_path = "data/ticket-sales.db"
                output_path = "data/ticket-sales-gold.txt"

                # Connect to the SQLite database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Query to calculate total sales for 'Gold' tickets
                cursor.execute("""
                    SELECT SUM(units * price)
                    FROM tickets
                    WHERE type = 'Gold'
                """)

                # Fetch result
                result = cursor.fetchone()[0]

                # If there are no "Gold" tickets, result might be None
                total_sales = result if result is not None else 0

                # Write the result to the output file
                with open(output_path, "w") as file:
                    file.write(str(total_sales))

                return {"status": "Gold ticket sales calculated", "total_sales": total_sales}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error calculating Gold ticket sales: {str(e)}")

            finally:
                if 'conn' in locals():
                    conn.close()
        return {"status": "Task not recognized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/read")
def read_file(path: str):
    file_path = f"./{path}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
