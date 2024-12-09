from fastapi import APIRouter
import requests

router = APIRouter()

API_KEY = '326e9c1c0c1b40148da5939d3d2117ff'

def get_waste_articles():
    url = f'https://newsapi.org/v2/everything?q=sampah&apiKey={API_KEY}&language=id'
    response = requests.get(url)
    data = response.json()

    if data["status"] == "ok" and data['articles']:
        return [
            {
                "title": article['title'],
                "image": article.get('urlToImage', ''),
                "description": article['description'],
                "url": article['url'],
            } for article in data['articles']
        ]
    else:
        return {"error": "Unable to fetch articles or no articles found"}

@router.get("/articles")
async def articles():
    articles = get_waste_articles()
    if not articles or "error" in articles:
        return {"message": "No articles found related to waste or error occurred"}
    return {"articles": articles}
