"""Turn raw URL features into short, human-readable warning signals.

These feed both the offline explanation and the LLM prompt, so the
explanation is grounded in what the feature extractor actually saw.
"""

from __future__ import annotations

from .features import FEATURE_NAMES, extract_features

_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def triggered_signals(url: str) -> list[str]:
    f = extract_features(url)

    def g(name: str) -> float:
        return float(f[_IDX[name]])

    signals: list[str] = []
    if g("ip"):
        signals.append("the host is a raw IP address instead of a domain name")
    if g("suspecious_tld"):
        signals.append("it uses a top-level domain often abused for throwaway sites (.tk, .xyz, .top ...)")
    if g("shortening_service"):
        signals.append("it goes through a URL-shortening service that hides the real destination")
    if g("nb_at"):
        signals.append("it contains an '@' character, which can disguise the real host")
    if g("punycode"):
        signals.append("the host uses punycode (xn--), a trick for look-alike characters")
    if g("phish_hints"):
        signals.append("it contains bait words such as secure, account, verify, login or update")
    if g("prefix_suffix"):
        signals.append("the hostname contains hyphens, common in look-alike domains")
    if g("abnormal_subdomain") or g("nb_subdomains") > 3:
        signals.append("it has an unusually deep chain of subdomains")
    if g("nb_redirection"):
        signals.append("another URL is embedded inside it (possible redirect)")
    if g("port"):
        signals.append("it specifies a non-standard port")
    if g("length_url") > 75:
        signals.append(f"it is long ({int(g('length_url'))} characters)")
    if g("ratio_digits_url") > 0.2:
        signals.append("a large share of the URL is digits")
    if not g("https_token"):
        signals.append("it does not use HTTPS (note: HTTPS alone does not prove a site is safe)")
    return signals
