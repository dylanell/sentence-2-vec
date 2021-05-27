"""
Construct a dataset of question-answer pairs on a singular topic from answers.com and write as a table to an SQLite DB.
"""


import yaml

from dataset.answers_topic_scraper import AnswersTopicScraper


def main():
    with open('.dataset_conf.yml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    
    scraper = AnswersTopicScraper(config['db_path'])

    scraper.scrape('fashion', 30000)

    scraper.close()


if __name__ =="__main__":
    main()