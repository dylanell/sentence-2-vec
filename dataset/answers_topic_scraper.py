"""
SQLite DB helper class.
"""


import gensim
import sqlite3
import requests
from bs4 import BeautifulSoup

from dataset.text_utils import test_qa_is_good, process_text


class AnswersTopicScraper():
    def __init__(self, write_path: str):
        """Initialize an Answers.com topic scraper to collect question-anwser 
        pairs, process the data, and write it to a sqlite database topic table.

        Args:
            write_path: path to write output files.

        """

        self._write_path = write_path
        self._db = sqlite3.connect(f'{self._write_path}/answers.db')
        self._cur = self._db.cursor()
        self._vocab = gensim.corpora.Dictionary()

        # add <UNK> token at index 0 for unknown word default
        self._vocab.add_documents([['<PAD>', '<UNK>']])

    def close(self):
        self._cur.close()
        self._db.close()

    def scrape(self, topic: str, min_samples: int):
        """Scrape up to min_samples question-answer pairs from an Answers.com 
        topic and write the data to the database.

        Args:
            topic: Answers.com topic from which to collect samples.
            min_samples: Number of samples to collect, or less if end of topic 
            content is reached.

        """

        self._cur.execute(f'''CREATE TABLE IF NOT EXISTS {topic}
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            url varchar(255) NOT NULL,
            question varchar(255) NOT NULL,
            answer varchar(255) NOT NULL,
            proc_question varchar(255) NOT NULL,
            proc_answer varchar(255) NOT NULL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')

        topic_url = f'https://www.answers.com/t/{topic}'
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
                {'class': 
                    'grid grid-cols-1 cursor-pointer justify-start '\
                    'items-start qCard my-4 p-4 bg-white md:rounded '\
                    'shadow-cardGlow'
                }
            )

            # check if questions on this page, otherwise break loop
            if len(question_divs) > 0:
                for i, block in enumerate(question_divs):
                    # extract question and answer blocks
                    q_block = block.find('h1', {'property': 'name'})
                    a_block = block.find('div', {'property': 'content'})

                    # if good data, add data to db and update vocabulary
                    if test_qa_is_good(q_block, a_block):
                        url = block.find('a')['href']
                        question = q_block.text
                        answer = a_block.text
                        proc_question = process_text(question)
                        proc_answer = process_text(answer)

                        try:
                            self._cur.execute(
                                f'''INSERT INTO {topic} 
                                (id, url, question, answer, proc_question, proc_answer) 
                                VALUES(?, ?, ?, ?, ?, ?)''', 
                                (num_samples, url, question, answer, proc_question, proc_answer)
                            )
                        except sqlite3.IntegrityError:
                            pass

                        self._vocab.add_documents((
                            proc_question.split(), proc_answer.split()))

                        num_samples += 1

                page_num += 1

            else:
                print('[INFO]: end of content')
                break

        self._db.commit()

        self._vocab.save(f'{self._write_path}/{topic}.vocab')