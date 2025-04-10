from time import sleep
from pydantic import BaseModel, Field
import substack_api
from bs4 import BeautifulSoup
from typing import List

class SubstackConfig(BaseModel):
    """Конфиги для SubstackProcessor."""
    target_substack: str = Field('https://gonzoml.substack.com/', description="Целевой Substack")
    query: str = Field(..., description="Запрос для поиска")
    limit: int = Field(3, ge=1, description="Ограничение количества постов на одну новостную ленту")

class SubstackProcessor:
    """Обработчик Substack."""
    
    def __init__(self, config: SubstackConfig):
        self.config = config
        self.newsletters = []
        self.posts = []
        
        target_newsletter = substack_api.Newsletter(config.target_substack)
        authors = target_newsletter.get_authors()
        if authors:
            author = authors[0]
            self.newsletters.append(target_newsletter)
            for sub in author.get_subscriptions():
                self.newsletters.append(substack_api.Newsletter('https://' + sub['domain']))

    def wide_search(self) -> None:
        """Широкий поиск по запросу."""
        self.posts.clear()  # Очистка предыдущих результатов
        for newsletter in self.newsletters:
            try:
                self.posts.extend(
                    newsletter.search_posts(
                        self.config.query, self.config.limit)
                )
            except Exception as e:
                print(f"Ошибка доступа к {newsletter}: {e}")
                continue

    def get_documents(self) -> List[str]:
        """Получение текстовых документов из постов."""
        documents = []
        for post in self.posts:
            try:
                content = post.get_content()
                if content:
                    documents.append(BeautifulSoup(content).get_text())
                    sleep(1)  # Задержка для избежания перегрузки API
            except Exception as e:
                print(f"Не удается загрузить страницу {post}: {e}")
                continue
        return documents

    def update_query(self, new_query: str):
        """Обновление запроса и сброс результатов."""
        new_config = SubstackConfig(
            target_substack=self.config.target_substack, 
            query=new_query, 
            limit=self.config.limit)
        self.config = new_config
        self.posts.clear()