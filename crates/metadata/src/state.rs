use chrono::{DateTime, Timelike, Utc};
use serde_json::{Map, Value, json};
use types::{Account, Tweet};

/// Lays out the `state` of metadata version `v2` to `v5`. `level` is the number after the `v`.
/// Never reads the clock: the account age is the time between the join date and the tweet.
pub fn object(level: u8, tweet: &Tweet, account: &Account, recent_tweets: &[Tweet]) -> Value {
    let age_days = account_age_days(tweet, account);
    let mut account_fields = Map::new();
    account_fields.insert("age_days".into(), json!(age_days));
    account_fields.insert("followers".into(), json!(account.followers));
    account_fields.insert("following".into(), json!(account.following));
    account_fields.insert("tweet_count".into(), json!(account.tweet_count));
    let favourites_count = raw_i64(&account.raw, "favouritesCount");
    account_fields.insert("favourites_count".into(), json!(favourites_count));
    account_fields.insert(
        "media_count".into(),
        json!(raw_i64(&account.raw, "mediaCount")),
    );
    account_fields.insert("blue_check".into(), json!(account.is_verified));
    account_fields.insert("x_automated_label".into(), json!(account.is_automated));
    if level >= 3 {
        account_fields.insert(
            "rates".into(),
            json!({
                "tweets_per_day": per_day(account.tweet_count.map(f64::from), age_days),
                "favourites_per_day": per_day(favourites_count.map(|n| n as f64), age_days),
                "followers_per_following": ratio(
                    account.followers.map(f64::from),
                    account.following.map(f64::from),
                ),
            }),
        );
    }
    if level >= 5 {
        account_fields.insert("name".into(), json!(raw_str(&account.raw, &["name"])));
        account_fields.insert(
            "bio".into(),
            json!(raw_str(&account.raw, &["profile_bio", "description"])),
        );
        account_fields.insert(
            "website".into(),
            json!(raw_str(
                &account.raw,
                &[
                    "profile_bio",
                    "entities",
                    "url",
                    "urls",
                    "0",
                    "expanded_url"
                ]
            )),
        );
        account_fields.insert(
            "location".into(),
            json!(raw_str(&account.raw, &["location"])),
        );
    }
    let mut state = Map::new();
    state.insert("tweet_text".into(), json!(tweet.text));
    state.insert("account".into(), Value::Object(account_fields));
    if level >= 4 {
        state.insert("recent_tweets".into(), recent_summary(recent_tweets));
    }
    Value::Object(state)
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn account_age_days(tweet: &Tweet, account: &Account) -> Option<f64> {
    let seconds = (tweet.posted_at? - account.joined_at?).num_seconds();
    Some(round2(seconds as f64 / 86_400.0))
}

fn per_day(count: Option<f64>, age_days: Option<f64>) -> Option<f64> {
    count
        .zip(age_days)
        .and_then(|(count, age)| divide(count, age))
}

fn ratio(top: Option<f64>, bottom: Option<f64>) -> Option<f64> {
    top.zip(bottom)
        .and_then(|(top, bottom)| divide(top, bottom))
}

fn divide(top: f64, bottom: f64) -> Option<f64> {
    (bottom > 0.0).then(|| round2(top / bottom))
}

fn raw_i64(raw: &Value, key: &str) -> Option<i64> {
    raw.get(key)?.as_i64()
}

/// Follows `path` through objects and arrays. An empty string counts as missing.
fn raw_str<'a>(raw: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut value = raw;
    for key in path {
        value = match key.parse::<usize>() {
            Ok(index) => value.get(index)?,
            Err(_) => value.get(key)?,
        };
    }
    value.as_str().filter(|text| !text.is_empty())
}

fn recent_summary(recent_tweets: &[Tweet]) -> Value {
    let mut posted: Vec<DateTime<Utc>> = recent_tweets
        .iter()
        .filter_map(|tweet| tweet.posted_at)
        .collect();
    let hours: Vec<u32> = posted.iter().map(|time| time.hour()).collect();
    posted.sort();
    let minutes_between: Vec<f64> = posted
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).num_seconds() as f64 / 60.0)
        .collect();
    json!({
        "count": recent_tweets.len(),
        "posting_hours_utc": (!recent_tweets.is_empty()).then_some(hours),
        "median_minutes_between": median(minutes_between).map(round2),
        "duplicate_text_share": duplicate_share(recent_tweets).map(round2),
    })
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

/// Lower case, collapsed whitespace, no URLs.
fn normalize(text: &str) -> String {
    text.split_whitespace()
        .filter(|word| !word.starts_with("http://") && !word.starts_with("https://"))
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The share of tweets whose normalized text equals the text of another tweet.
fn duplicate_share(tweets: &[Tweet]) -> Option<f64> {
    if tweets.is_empty() {
        return None;
    }
    let texts: Vec<String> = tweets.iter().map(|tweet| normalize(&tweet.text)).collect();
    let duplicates = texts
        .iter()
        .filter(|text| texts.iter().filter(|other| other == text).count() > 1)
        .count();
    Some(duplicates as f64 / texts.len() as f64)
}
