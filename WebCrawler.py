from icrawler.builtin import GoogleImageCrawler

for keyword in ['cat','dog']:
    google_crawler=GoogleImageCrawler(
        parser_threads=2,
        downloader_threads=4,
        storage={'root_dir':'images/{}'.format(keyword)}
    )

    google_crawler.crawl(
        keyword=keyword,max_num=500,min_size=(100,100))