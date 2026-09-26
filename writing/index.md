---
layout: default
title: Writing
permalink: /writing/
---

<h1>Writing</h1>

<p>Recent posts and essays.</p>

<ul>
  {% for post in site.posts %}
    <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small class="small">({{ post.date | date: "%Y-%m-%d" }})</small></li>
  {% endfor %}
</ul>
