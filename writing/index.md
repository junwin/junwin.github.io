---
layout: default
title: Writing
permalink: /writing/
---

<h1>Writing</h1>

<p>Recent posts and essays.</p>

<div class="post-grid">
  {% for post in site.posts %}
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
.mindfulness-card h3{font-size:1rem;line-height:1.3;font-weight:inherit}
.mindfulness-card .mindfulness-date{margin:0;padding:0 .5rem .5rem}
.mindfulness-card img{height:160px}
</style>
