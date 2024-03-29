"""
Construct a dataset of question-answer pairs on a singular topic from answers.com and write as a table to an SQLite DB.
"""


import yaml

from scraper.answers_topic_scraper import AnswersTopicScraper


def main():
    with open('config/scrape_cfg.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    scraper = AnswersTopicScraper(config['write_path'])

    scraper.scrape(config['topic'], config['max_samples'])

    scraper.close()


if __name__ =="__main__":
    main()