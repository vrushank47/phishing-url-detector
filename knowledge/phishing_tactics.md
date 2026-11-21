# Raw IP address in the host
Legitimate services almost always use a registered domain name. A URL whose host is a bare IP address, such as http://192.0.2.10/login, often points to a temporary server set up by an attacker. Be cautious about entering credentials on such pages.

# Suspicious top-level domains
Some TLDs such as .tk, .ml, .ga, .cf, .gq, .xyz and .top are free or very cheap, so attackers register many disposable domains there. A cheap TLD is not proof of phishing, but it combined with bait words in the URL is a strong warning sign.

# Bait keywords: login, verify, secure, account
Phishing URLs often stuff words like secure, account, verify, login, update, signin or banking into the hostname or path to look official. Real companies normally host these pages on their main domain, not on a long domain made of keywords.

# Hyphenated look-alike domains
Domains such as paypal-secure-login.example.com imitate a brand by adding hyphens and extra words. The part that matters is the registered domain just before the TLD. Everything to its left is controlled by whoever owns that domain.

# Deep subdomain chains
Attackers place a trusted name in a subdomain, for example accounts.google.com.attacker.example. The real owner of the site is the domain at the right end, here attacker.example. Many dots in the hostname is a common trick to push the real domain out of view.

# The @ symbol trick
In a URL, anything before an @ in the authority part is treated as user information and ignored when choosing the server. https://www.bank.com@evil.example/ actually connects to evil.example. Any @ in a link is worth a closer look.

# URL shorteners
Services like bit.ly or tinyurl hide the destination. Attackers use them to bypass visual inspection and filters. Expand a short link with a preview tool before opening it, and do not enter credentials on a page reached through an unexpected short link.

# Punycode and homograph attacks
Hostnames starting with xn-- are encoded international characters. Attackers use look-alike letters from other alphabets, such as a Cyrillic a in place of a Latin a, to build domains that look identical to a real brand.

# Embedded redirect URLs
A URL that contains another http:// or https:// address in its path or query can be an open redirect: the trusted site forwards the victim to the attacker's page. Be wary when a link to a known site carries a second URL inside it.

# HTTPS is not proof of safety
A padlock only means the connection is encrypted, not that the site is honest. Many phishing sites use valid HTTPS certificates. Judge the domain name, not the padlock.

# Very long or heavily encoded URLs
Long URLs with many digits, percent-encoded characters or random-looking strings can hide the real destination or carry tracking and session data used in credential-harvesting kits.

# What to do if you suspect phishing
Do not enter passwords or card details. Do not click further. Open the service by typing its official address yourself or using a saved bookmark. Report the message to your IT or security team or the impersonated company. If you already entered credentials, change that password immediately and anywhere else it was reused, and enable multi-factor authentication.

# Limits of URL-only detection
A model that looks only at the URL string cannot see page content, certificates or domain age. It can miss phishing pages hosted on compromised legitimate sites and can flag unusual but harmless URLs. Treat its output as advisory and verify important cases by hand.
