from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
#crawler = GoogleImageCrawler(downloader_threads=4, storage={"root_dir": "paru"})
crawler = BingImageCrawler(downloader_threads=4,storage={"root_dir": "dog"})
crawler.crawl(keyword="dog", max_num=500)

crawler = BingImageCrawler(downloader_threads=4,storage={"root_dir": "cat"})
crawler.crawl(keyword="cat", max_num=500)