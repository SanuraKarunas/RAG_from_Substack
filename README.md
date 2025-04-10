# RAG_from_Substack
## RAG-система с использованием статей из Substack'а по необходимой тематике.
- токенайзер: sentence-transformers/all-mpnet-base-v2
- языковая модель: google/flan-t5-xl (отсутствует железо, чтобы подключать более крупные языковые модели из transformers)
- векторная база данных: Faiss
- промпты, схемы, прочее: langchain
- парсинг статей с Substack: substack_api, beautifulsoup

Демо:
![Demo](demo.mp4)

## Процесс:
1. Парсинг статей с substack
   - Берется таргетный канал на Substack (в данном случае 'https://gonzoml.substack.com/'), находится автор канала
   - Cобираются каналы, на которые тот подписан (таким образом мы смотрим каких эекспертов читает наш эксперт)
   - На этих каналах происходит поиск статей по ключевому слову (в данном случае 'deepseek')
   - Найденные статьи парсятся
   
2. Создание векторной базы данных
   - Статьи проходят предобработку, создается List[Documents]
   - Создается векторная база данных на основе листа

3. RAG-система
   - Подгружается модель
   - Происходит обработка запроса, на его основе LLM формулируют промпты для генерации поисковых запросов
   - В векторной базе данных находятся наиболее релевантные документы, они подгружаются к запросу LLM в качестве контекста
   - Модель выдает свой ответ

4. Телеграм бот
   - Взаимодействие происходит через телеграм бот. Имеется возможность ввести запрос без использования RAG.

## Что можно доделать?
- Сохранять в проекте последнюю загруженную векторную базу данных вместе с документами, тогда перед каждым включением бота не придется ждать 10 минут пока спарсятся данные
- Интегрировать в чатбот поддержку функционала "сделать новый запрос по Substack"
- Вместе того, чтобы брать каналы, на которые подписан эксперт, лучше брать заранее продуманный список, иначе можно кого-то пропустить или зря слушать
- Добавить метаданные относительно статьи и канала, на котором ее запостили
- Сделать более тщательную предобработку текста, авторы статей на substack слишком часто используют эмодзи
- Попробовать искать не по substack, а по статьям, которые цитируют в твиттере или медиуме
- Поиграться с промптами и конфигурацией модели, подключаться к внешним вычислительным мощностям в целях улучшения модели    
   
