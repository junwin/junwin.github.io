---
layout: default
title: Mindfulness
permalink: /mindfulness/
---

# 🌿 Mindfulness

Reflections on mindful breathing and living in the present moment, drawing on
the teachings of Thich Nhat Hanh and the Plum Village tradition.

<div class="post-grid mindfulness-grid">
  {% assign mindful_posts = site.posts | where_exp: "post", "post.categories contains 'mindfulness'" | sort: "date" | reverse %}
  {% for post in mindful_posts %}
    <a class="card mindfulness-card" href="{{ post.url }}">
      {% if post.image %}
      <img src="{{ post.image }}" alt="{{ post.title | escape }}" loading="lazy">
      {% endif %}
      <h3>{{ post.title }}</h3>
      <p class="small mindfulness-date">{{ post.date | date: "%Y-%m-%d" }}</p>
    </a>
  {% endfor %}
</div>

<style>
.mindfulness-grid{margin-top:1rem}
.mindfulness-card h3{font-size:1rem;line-height:1.3;font-weight:inherit}
.mindfulness-card .mindfulness-date{margin:0;padding:0 .5rem .5rem}
.mindfulness-card img{height:160px}
</style>
