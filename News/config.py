# config.py
from datetime import datetime

# API Key (should be stored in secrets, but included for completeness)
API_KEY = "AIzaSyC2Ocy2OgfbgaHaX5xQzxLhDsb_ioAW6pA"

# Current date strings
CURRENT_DATE = datetime.now().strftime("%B %d, %Y")
CURRENT_DATE_SHORT = datetime.now().strftime("%b %d, %Y").upper()

# Reliable RSS Feeds
NEWS_FEEDS = {
    "Philippine News": {
        "Inquirer Headlines":  "https://www.inquirer.net/fullfeed",
        "Philstar News":       "https://www.philstar.com/rss/headlines",
        "GMA News":            "https://data.gmanetwork.com/gno/rss/news/feed.xml",
        "Manila Bulletin":     "https://mb.com.ph/rss",
        "Manila Times":        "https://www.manilatimes.net/news/feed",
        "Rappler":             "https://www.rappler.com/feed/",
        "ABS-CBN News":        "https://news.abs-cbn.com/rss/Headlines.xml",
        "BusinessWorld PH":    "https://www.bworldonline.com/feed/",
        "Inquirer Business":   "https://business.inquirer.net/feed",
        "SunStar Network":     "https://www.sunstar.com.ph/rss/latest",
    },
    "International News": {
        "BBC World":           "http://feeds.bbci.co.uk/news/world/rss.xml",
        "Reuters World":       "https://feeds.reuters.com/reuters/worldNews",
        "Al Jazeera":          "https://www.aljazeera.com/xml/rss/all.xml",
        "The Guardian World":  "https://www.theguardian.com/world/rss",
        "NYT World":           "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "CNN Top Stories":     "http://rss.cnn.com/rss/edition.rss",
        "AP Top News":         "https://feeds.apnews.com/rss/apf-topnews",
        "DW World":            "https://rss.dw.com/rdf/rss-en-world",
        "Foreign Policy":      "https://foreignpolicy.com/feed/",
        "UN News":             "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
    },
}