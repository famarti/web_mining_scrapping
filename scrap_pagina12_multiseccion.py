import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from twisted.internet import reactor, defer
from urllib import parse
from os import path
from scrapy.http.response.html import HtmlResponse
from typing import List


class NewsSpider(CrawlSpider):
    name = 'crawler_pagina12'
    allowed_domains = ['pagina12.com.ar']
    
    rules = (
        Rule(LinkExtractor(allow=r'/\d{6,}', deny_domains=['auth.pagina12.com.ar'],
                           deny_extensions=['7z', '7zip', 'apk', 'bz2', 'cdr,' 'dmg', 'ico,' 'iso,' 'tar', 'tar.gz', 'pdf', 'docx', 'jpg', 'png', 'css', 'js']),
             callback='parse_response', follow=True),
    )
     
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
        'LOG_ENABLED': True,
        'LOG_LEVEL': 'INFO',
        'DEPTH_LIMIT': 2,
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 3,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
    }

    def __init__(self, save_pages_in_dir='.', max_items=15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.basedir = save_pages_in_dir
        self.max_items = max_items
        self.items_downloaded = 0
    
    def parse_response(self, response: HtmlResponse):
        if self.items_downloaded >= self.max_items:
            self.crawler.engine.close_spider(self, 'Límite de artículos alcanzado')
            return

        html_filename = path.join(self.basedir, parse.quote(response.url[response.url.rfind("/") + 1:]))
        if not html_filename.endswith(".html"):
            html_filename += ".html"
        print("URL:", response.url, "Página guardada en:", html_filename)

        with open(html_filename, "wt", encoding="utf8") as html_file:
            html_file.write(response.body.decode("utf8"))

        self.items_downloaded += 1


@defer.inlineCallbacks
def crawl():
    DIR_EN_DONDE_GUARDAR_PAGINAS = "pagina_12_noticias"
    secciones = ["sociedad", "el-mundo", "el-pais", "economia"]
    max_items_per_section = 40
    
    for seccion in secciones:
        url = f"https://www.pagina12.com.ar/secciones/{seccion}?page=1"
        save_dir = f"{DIR_EN_DONDE_GUARDAR_PAGINAS}/{seccion}"
        runner = CrawlerRunner()
        yield runner.crawl(NewsSpider, save_pages_in_dir=save_dir, start_urls=[url], max_items=max_items_per_section)
    
    reactor.stop()

configure_logging()
crawl()
reactor.run()  # Inicia el reactor de Twisted
