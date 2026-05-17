from __future__ import annotations

import pytest

from agents.llm_tools import parse_event_with_stub_llm_or_rules
from agents.router_forecast_agent import RouterForecastAgent
from agents.templates import (
    COMPANY_TECH_ANNOUNCEMENT,
    CULTURE_AWARDS_ENTERTAINMENT,
    GENERIC_NEWS_UNIQUE,
    MACRO_RELEASE,
    POLITICS_ELECTIONS_POLICY,
    PRICE_THRESHOLD,
    SPORTS,
    WEATHER,
)


ROUTE_CASES = [
    ("Will BTC exceed $100,000 by June 30?", "Financials", PRICE_THRESHOLD),
    ("Will Bitcoin be below $80,000 on Friday?", "Crypto", PRICE_THRESHOLD),
    ("Will ETH close above $4,000 this month?", "Crypto", PRICE_THRESHOLD),
    ("Will gold exceed $3,000 before July?", "Financials", PRICE_THRESHOLD),
    ("Will the Nasdaq 100 close above 22,000 tomorrow?", "Financials", PRICE_THRESHOLD),
    ("Will Tesla shares trade below $150 this week?", "Financials", PRICE_THRESHOLD),
    ("Will crude oil settle above $90 by month end?", "Financials", PRICE_THRESHOLD),
    ("Will EUR/USD exceed 1.10 by Friday?", "Financials", PRICE_THRESHOLD),
    ("Will US CPI year-over-year be above 3.0% for May?", "Economics", MACRO_RELEASE),
    ("Will core inflation come in under 0.2% month-over-month?", "Economics", MACRO_RELEASE),
    ("Will the Fed cut rates at the June FOMC meeting?", "Economics", MACRO_RELEASE),
    ("Will unemployment be at least 4.5% in the next jobs report?", "Economics", MACRO_RELEASE),
    ("Will GDP growth exceed 2% in Q2?", "Economics", MACRO_RELEASE),
    ("Will 10-year Treasury yield be above 5% on Friday?", "Financials", MACRO_RELEASE),
    ("Will the Bank of Japan hike rates in June?", "Economics", MACRO_RELEASE),
    ("Will the Lakers beat the Celtics tonight?", "Sports", SPORTS),
    ("Will Yankees win their MLB game?", "Sports", SPORTS),
    ("Will Arsenal win the Premier League title?", "Sports", SPORTS),
    ("Will Felix Auger-Aliassime beat Vit Kopriva?", "Sports", SPORTS),
    ("Sho Shimabukuro vs Stefanos Sakellaridis: Total Games", "Sports", SPORTS),
    ("Will RED Academy win map 2 in the League of Legends match?", "Sports", SPORTS),
    ("Will over 45.5 points be scored in the NBA game?", "Sports", SPORTS),
    ("Will McIlroy win the Masters?", "Sports", SPORTS),
    ("Will UFC fighter A beat fighter B?", "Sports", SPORTS),
    ("Will New York City exceed 95F on July 4?", "Climate and Weather", WEATHER),
    ("Will Miami receive over 2 inches of rainfall tomorrow?", "Climate and Weather", WEATHER),
    ("Will a hurricane make landfall in Florida this season?", "Climate and Weather", WEATHER),
    ("Will snowfall in Boston exceed 6 inches this week?", "Climate and Weather", WEATHER),
    ("Will the world pass 2 degrees Celsius over pre-industrial levels before 2050?", "Climate and Weather", WEATHER),
    ("Will Chicago temperature be below 20 degrees on Christmas?", "Climate and Weather", WEATHER),
    ("Will Trump win the 2028 presidential election?", "Elections", POLITICS_ELECTIONS_POLICY),
    ("Will Democrats win control of the Senate?", "Elections", POLITICS_ELECTIONS_POLICY),
    ("Will Congress pass the stablecoin bill before August?", "Politics", POLITICS_ELECTIONS_POLICY),
    ("Who will be the next Pope? - YES side: Luis Antonio Tagle", "Elections", POLITICS_ELECTIONS_POLICY),
    ("Will Zohran Mamdani become President of the United States before 2045?", "Elections", POLITICS_ELECTIONS_POLICY),
    ("Will Li Ganjie be named Xi Jinping successor?", "Elections", POLITICS_ELECTIONS_POLICY),
    ("Will OpenAI announce GPT-6 before December?", "Technology", COMPANY_TECH_ANNOUNCEMENT),
    ("Will Apple launch a foldable iPhone this year?", "Technology", COMPANY_TECH_ANNOUNCEMENT),
    ("Will Tesla report positive earnings next quarter?", "Companies", COMPANY_TECH_ANNOUNCEMENT),
    ("Will Nvidia announce a new AI chip at Computex?", "Technology", COMPANY_TECH_ANNOUNCEMENT),
    ("Will SpaceX conduct a Starship launch before July?", "Science and Technology", COMPANY_TECH_ANNOUNCEMENT),
    ("Will Oppenheimer win Best Picture at the Oscars?", "Entertainment", CULTURE_AWARDS_ENTERTAINMENT),
    ("Will Taylor Swift top the Billboard Hot 100 next week?", "Entertainment", CULTURE_AWARDS_ENTERTAINMENT),
    ("Will Eurovision be won by Sweden?", "Entertainment", CULTURE_AWARDS_ENTERTAINMENT),
    ("Will an anime film win the box office this weekend?", "Entertainment", CULTURE_AWARDS_ENTERTAINMENT),
    ("Will the Grammy for Album of the Year go to Billie Eilish?", "Entertainment", CULTURE_AWARDS_ENTERTAINMENT),
    ("Will Elon Musk visit Mars before 2099?", "World", GENERIC_NEWS_UNIQUE),
    ("Will humans colonize Mars before 2050?", "Science and Technology", GENERIC_NEWS_UNIQUE),
    ("Will California high-speed rail begin public service before 2030?", "Transportation", GENERIC_NEWS_UNIQUE),
    ("Will Wikipedia be inaccessible for more than 24 hours this year?", "World", GENERIC_NEWS_UNIQUE),
]


@pytest.mark.parametrize(("title", "category", "expected_template"), ROUTE_CASES)
def test_synthetic_question_routing(title: str, category: str, expected_template: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TEMPLATE_ROUTE_LLM_VERIFY", raising=False)
    event = {"title": title, "category": category, "outcomes": ["Yes", "No"]}
    spec = parse_event_with_stub_llm_or_rules(event)

    route = RouterForecastAgent().route_template(event, spec)

    assert route.template_name == expected_template

