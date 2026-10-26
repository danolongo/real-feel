# Spike 2.1: X data source

Date: 2026-10-25. Result: X blocks self-hosted Firecrawl. The scraper uses twitterapi.io. This settles O1.

## Self-hosted Firecrawl

You ran Firecrawl `v2.11.162` from its prebuilt images, with Workers AI as its language model.

| Request | Result |
|---------|--------|
| `x.com/search?q=…` | Failed |
| A profile page | Failed |
| A tweet page | Failed |
| `/v2/search` with `site:x.com …` | 10 Google results with titles and snippets. No tweets and no IDs. |

- Firecrawl's Playwright engine loaded each X page. Firecrawl then logged "deemed unsuccessful due to proxy inadequacy" and asked for its stealth proxy. Only Fire-engine has that proxy, and self-hosted Firecrawl lacks Fire-engine.
- `/v2/search` asks Google, not X. Firecrawl then tried to scrape each X result, and X blocked those pages too.
- Firecrawl's JSON extraction calls the OpenAI `/v1/responses` API. Workers AI rejects that request with `400 invalid_prompt`.
- Building Firecrawl from source needs more than 2 GiB of memory and about 30 GB of disk. The prebuilt images avoid the build.

Logged-out visitors cannot search X. Every local scraper that still works (twscrape, Twikit) logs in with X accounts. That breaks X's terms and gets the accounts banned.

## twitterapi.io

`scripts/spike-twitterapi.sh` calls three endpoints. All three succeed. The table also lists `last_tweets`, which the documentation describes and no call has tested.

| Endpoint | Returns |
|----------|---------|
| `GET /twitter/tweet/advanced_search` | 20 tweets per page and a `next_cursor` |
| `GET /twitter/user/info` | One account |
| `GET /twitter/tweets?tweet_ids=` | Up to 50 tweets by ID |
| `GET /twitter/user/last_tweets` (not yet called live) | Up to 20 recent tweets of one account per page, with `data.tweets`, `has_next_page`, and `next_cursor` |

Every tweet in a search result carries its author, so the scraper needs one call for each page and no profile calls.

**Tweet fields:** `id`, `text`, `createdAt`, `lang`, `isReply`, `inReplyToId`, `conversationId`, `likeCount`, `replyCount`, `retweetCount`, `quoteCount`, `viewCount`, `bookmarkCount`, `entities`, `source`, `quoted_tweet`, `retweeted_tweet`.

**Author fields:** `id`, `userName`, `name`, `description`, `location`, `createdAt`, `followers`, `following`, `statusesCount`, `favouritesCount`, `mediaCount`, `isBlueVerified`, `isVerified`, `verifiedType`, `isAutomated`, `automatedBy`, `protected`, `profilePicture`, `pinnedTweetIds`.

- **IDs:** a tweet ID and an account ID hold X's numeric IDs as strings. They never change, so upserts stay idempotent.
- **Dates:** search results use X's format `Sun Oct 25 09:56:57 +0000 2026`. `/user/info` uses ISO 8601.
- **`isAutomated`:** X's own label for automated accounts. All 20 authors in the sample had `false`.
- **Price:** $0.15 per 1,000 tweets, with at least $0.00015 per call. New accounts get up to $1 of credit.
- **Rate limit:** the free tier allows one request every 5 seconds. A second request inside that window returns `429`.

## Alternatives

| Source | Cost per 1,000 tweets | Why not |
|--------|-----------------------|---------|
| Official X API | about $5 | 30 times the price |
| Apify Tweet Scraper | $0.25 to $0.40 | The free tier has no API access |
| SocialData | $0.20 | Similar, a little more expensive |
| SocialCrawl | not published for X | One schema across 69 platforms, and its X fields are unconfirmed |
| A local scraper with X accounts | free | Breaks X's terms, risks bans, breaks when X rotates endpoints |

Most prices come from vendor comparison pages. Check the provider's own pricing page before you rely on one.
