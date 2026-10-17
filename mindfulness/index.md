---
layout: default
title: Mindfulness
permalink: /mindfulness/
---

# 🌿 Mindfulness

Reflections on mindful breathing and living in the present moment, drawing on
the teachings of Thich Nhat Hanh and the Plum Village tradition.

<ul>
  {% assign mindful_posts = site.posts | where_exp: "post", "post.categories contains 'mindfulness'" | sort: "date" | reverse %}
  {% for post in mindful_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>({{ post.date | date: "%Y-%m-%d" }})</small>
    </li>
  {% endfor %}
</ul>
