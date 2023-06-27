from neo4j import GraphDatabase
from argparse import ArgumentParser
import pandas as pd


def get_metadata(driver: GraphDatabase.driver, db: str, metadata: list) -> list:
    q = [
        f'match (a:Artwork)--({v["var"]}: {v["class"].capitalize()})' for v in metadata
    ]
    q = '\n'.join(q)
    q += '\nreturn a.name as artwork, '
    q += ', '.join([
        f'{v["var"]}.name as {v["class"]}' for v in metadata
    ])

    with driver.session(database=db) as session:
        return session.run(q).data()


def get_conf():
    argparser = ArgumentParser()
    argparser.add_argument('--uri', type=str)
    argparser.add_argument('--db', type=str)
    argparser.add_argument('--outpath', type=str)
    return argparser.parse_args()


def launch():
    args = get_conf()
    driver = GraphDatabase.driver(uri=args.uri, auth=('neo4j', 'neo4j'))
    metadata = [
        {'var': 's', 'class': 'style'},
        {'var': 'g', 'class': 'genre'},
    ]
    data = get_metadata(driver, args.db, metadata)
    pd.DataFrame(data).to_json('metadata.json')


if __name__ == '__main__':
    launch()