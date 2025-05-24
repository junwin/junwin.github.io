---
layout: default
title: John Unwin's Ramblings
---

## Welcome

This is a personal collection of essays, notes, and thoughts on software, systems, and the occasional philosophical detour.

Here are some of the latest posts:

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>({{ post.date | date: "%Y-%m-%d" }})</small>
    </li>
  {% endfor %}
</ul>

---

### Other links

- [GitHub Profile](https://github.com/junwin)
- [Twitter](https://twitter.com/junwin)
