use chrono::{DateTime, TimeZone, Utc};
use metadata::build;
use serde_json::{Value, json};
use types::{Account, MetadataVersion, ModelInput, QuestionVersion, Tweet};

fn at(day: u32, hour: u32, minute: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 3, day, hour, minute, 0).unwrap()
}

fn tweet(id: &str, text: &str, posted_at: Option<DateTime<Utc>>) -> Tweet {
    Tweet {
        id: id.into(),
        account_id: "a1".into(),
        text: text.into(),
        posted_at,
        raw: json!({}),
    }
}

fn account() -> Account {
    Account {
        id: "a1".into(),
        handle: "maple_fox".into(),
        // 10 days and 12 hours before the tweet below.
        joined_at: Some(Utc.with_ymd_and_hms(2026, 3, 1, 0, 0, 0).unwrap()),
        followers: Some(30),
        following: Some(40),
        tweet_count: Some(105),
        is_verified: Some(true),
        is_automated: Some(false),
        raw: json!({
            "name": "Maple Fox",
            "favouritesCount": 21,
            "mediaCount": 3,
            "location": "Lisbon",
            "description": "",
            "profile_bio": {
                "description": "I post about maple trees.",
                "entities": { "url": { "urls": [{ "expanded_url": "https://maple.example" }] } },
            },
        }),
    }
}

fn main_tweet() -> Tweet {
    tweet("1", "Maple syrup season!", Some(at(11, 12, 0)))
}

fn run(metadata: &str, question: &str, account: &Account, recent_tweets: &[Tweet]) -> ModelInput {
    build(
        &main_tweet(),
        account,
        recent_tweets,
        &MetadataVersion(metadata.into()),
        &QuestionVersion(question.into()),
    )
    .unwrap()
}

fn recent() -> Vec<Tweet> {
    vec![
        tweet("2", "Good  MORNING https://t.co/a", Some(at(10, 8, 0))),
        tweet("3", "good morning", Some(at(10, 8, 30))),
        tweet("4", "Something else", Some(at(10, 10, 30))),
        tweet("5", "Syrup again", Some(at(10, 23, 0))),
    ]
}

fn v2_state() -> Value {
    json!({
        "tweet_text": "Maple syrup season!",
        "account": {
            "age_days": 10.5,
            "followers": 30,
            "following": 40,
            "tweet_count": 105,
            "favourites_count": 21,
            "media_count": 3,
            "blue_check": true,
            "x_automated_label": false,
        },
    })
}

fn v3_state() -> Value {
    let mut state = v2_state();
    state["account"]["rates"] = json!({
        "tweets_per_day": 10.0,
        "favourites_per_day": 2.0,
        "followers_per_following": 0.75,
    });
    state
}

fn v4_state() -> Value {
    let mut state = v3_state();
    state["recent_tweets"] = json!({
        "count": 4,
        "posting_hours_utc": [8, 8, 10, 23],
        "median_minutes_between": 120.0,
        "duplicate_text_share": 0.5,
    });
    state
}

fn v5_state() -> Value {
    let mut state = v4_state();
    state["account"]["name"] = json!("Maple Fox");
    state["account"]["bio"] = json!("I post about maple trees.");
    state["account"]["website"] = json!("https://maple.example");
    state["account"]["location"] = json!("Lisbon");
    state
}

#[test]
fn v2_state_is_exact() {
    assert_eq!(run("v2", "v1", &account(), &[]).state, v2_state());
}

#[test]
fn v3_state_is_exact() {
    assert_eq!(run("v3", "v1", &account(), &[]).state, v3_state());
}

#[test]
fn v4_state_is_exact() {
    assert_eq!(run("v4", "v1", &account(), &recent()).state, v4_state());
}

#[test]
fn v5_state_is_exact() {
    assert_eq!(run("v5", "v1", &account(), &recent()).state, v5_state());
}

#[test]
fn v2_and_v3_ignore_the_recent_tweets() {
    for version in ["v2", "v3"] {
        assert_eq!(
            run(version, "v1", &account(), &recent()),
            run(version, "v1", &account(), &[])
        );
    }
}

#[test]
fn missing_values_become_null_never_zero() {
    let empty = Account {
        joined_at: None,
        followers: None,
        following: None,
        tweet_count: None,
        is_verified: None,
        is_automated: None,
        raw: json!({}),
        ..account()
    };
    let state = run("v5", "v1", &empty, &[]).state;
    assert_eq!(
        state["account"],
        json!({
            "age_days": null,
            "followers": null,
            "following": null,
            "tweet_count": null,
            "favourites_count": null,
            "media_count": null,
            "blue_check": null,
            "x_automated_label": null,
            "rates": {
                "tweets_per_day": null,
                "favourites_per_day": null,
                "followers_per_following": null,
            },
            "name": null,
            "bio": null,
            "website": null,
            "location": null,
        })
    );
}

#[test]
fn a_zero_divisor_gives_a_null_rate() {
    let account = Account {
        following: Some(0),
        joined_at: Some(at(11, 12, 0)),
        ..account()
    };
    let state = run("v3", "v1", &account, &[]).state;
    assert_eq!(state["account"]["age_days"], json!(0.0));
    assert_eq!(
        state["account"]["rates"],
        json!({
            "tweets_per_day": null,
            "favourites_per_day": null,
            "followers_per_following": null,
        })
    );
}

#[test]
fn the_age_comes_from_the_tweet_time_not_the_clock() {
    let later = tweet("1", "Maple syrup season!", Some(at(21, 12, 0)));
    let input = build(
        &later,
        &account(),
        &[],
        &MetadataVersion("v2".into()),
        &QuestionVersion("v1".into()),
    )
    .unwrap();
    assert_eq!(input.state["account"]["age_days"], json!(20.5));
}

#[test]
fn an_empty_bio_counts_as_missing() {
    let mut account = account();
    account.raw["profile_bio"]["description"] = json!("");
    let state = run("v5", "v1", &account, &[]).state;
    assert_eq!(state["account"]["bio"], Value::Null);
}

#[test]
fn v4_and_v5_accept_an_empty_recent_tweets_slice() {
    for version in ["v4", "v5"] {
        let state = run(version, "v1", &account(), &[]).state;
        assert_eq!(
            state["recent_tweets"],
            json!({
                "count": 0,
                "posting_hours_utc": null,
                "median_minutes_between": null,
                "duplicate_text_share": null,
            })
        );
    }
}

#[test]
fn the_duplicate_share_ignores_case_spacing_and_urls() {
    let tweets = vec![
        tweet("2", "Buy NOW   https://t.co/x", None),
        tweet("3", "buy now", None),
        tweet("4", "buy now https://t.co/y", None),
        tweet("5", "other", None),
    ];
    let state = run("v4", "v1", &account(), &tweets).state;
    assert_eq!(state["recent_tweets"]["duplicate_text_share"], json!(0.75));
    assert_eq!(
        state["recent_tweets"]["median_minutes_between"],
        Value::Null
    );
}

#[test]
fn question_versions_hold_the_documented_questions() {
    let questions = |version: &str| -> Vec<(String, String)> {
        run("v1", version, &account(), &[])
            .questions
            .into_iter()
            .map(|q| (q.id, q.instructions))
            .collect()
    };
    let v2 = questions("v2");
    assert_eq!(v2.len(), 1);
    assert_eq!(v2[0].0, "bot");
    assert_eq!(
        v2[0].1,
        "Is this account run by software rather than by a person? Software includes scripts, schedulers, and LLM agents."
    );
    let v3 = questions("v3");
    assert_eq!(v3.len(), 1);
    assert_eq!(
        v3[0].1,
        "Is this account a bot? Count both scripted bots and AI agents: accounts that an LLM runs and that write their own fluent, varied tweets. A person who sometimes posts AI-written text is not a bot."
    );
    let v4 = questions("v4");
    assert_eq!(v4.len(), 2);
    assert_eq!(v4[0], v3[0]);
    assert_eq!(v4[1].0, "llm_text");
    assert_eq!(
        v4[1].1,
        "Did an LLM write the text of this tweet, whoever posted it?"
    );
}

#[test]
fn every_metadata_version_works_with_every_question_version() {
    for metadata in ["v1", "v2", "v3", "v4", "v5"] {
        for question in ["v1", "v2", "v3", "v4"] {
            let input = run(metadata, question, &account(), &recent());
            assert_eq!(input.questions[0].id, "bot");
        }
    }
}
