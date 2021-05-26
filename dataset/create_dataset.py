"""
Construct a dataset of question-answer pairs on a singular topic from answers.com and save to SQLite DB file.
"""

import yaml
import sqlite3
import requests
from bs4 import BeautifulSoup

import utils

def main():
    with open('.dataset_conf.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    db = sqlite3.connect(f'{config["db_path"]}')
    cursor = db.cursor()

    topic_url = config["topic_url"]
    topic = topic_url.split("/")[-1]
    min_samples = config["min_samples"]

    cursor.execute(f'''CREATE TABLE IF NOT EXISTS {topic}
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
        url varchar(255) NOT NULL,
        question varchar(255) NOT NULL,
        answer varchar(255) NOT NULL,
        proc_question varchar(255) NOT NULL,
        proc_answer varchar(255) NOT NULL,
        time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')

    page_num = 0
    num_samples = 0

    print('[INFO]: scraping pages for \'{}\''.format(topic_url))
    while num_samples < min_samples:
        # pull page content and parse
        headers = {'User-Agent': 'Mozilla/5.0'}
        page = requests.get(
            f'{topic_url}/best?page={page_num}', headers=headers)
        assert page.status_code == 200, 'page download unsuccessful'
        soup = BeautifulSoup(page.content, 'html.parser')

        print(f"[INFO]: scraping \'{soup.title.text}\', page: {page_num}")

        # extract all question divs from this page
        question_divs = soup.find_all(
            'div', 
            {"class": "grid grid-cols-1 cursor-pointer justify-start items-start qCard my-4 p-4 bg-white md:rounded shadow-cardGlow"}
        )

        # check if questions on this page, otherwise break loop
        if len(question_divs) > 0:
            for i, block in enumerate(question_divs):
                # extract question and answer blocks
                q_block = block.find('h1', {'property': 'name'})
                a_block = block.find('div', {'property': 'content'})

                # if good question and answer, add data to db
                if utils.test_qa_is_good(q_block, a_block):
                    url = block.find('a')['href']
                    question = q_block.text
                    answer = a_block.text
                    proc_question = utils.process_text(question)
                    proc_answer = utils.process_text(answer)

                    cursor.execute(
                        f'''INSERT INTO {topic} 
                        (id, url, question, answer, proc_question, proc_answer) 
                        VALUES(?, ?, ?, ?, ?, ?)''', 
                        (num_samples, url, question, answer, proc_question, proc_answer)
                    )

                    num_samples += 1

            page_num += 1

        else:
            print('[INFO]: end of content')
            break

    db.commit()
    cursor.close()
    db.close()

if __name__ =="__main__":
    main()