---
layout: default
title: Mindfulness
permalink: /mindfulness/
---

# 🌿 Mindfulness

Reflections on mindful breathing and living in the present moment, drawing on
the teachings of Thich Nhat Hanh and the Plum Village tradition.

<div class="mindfulness-grid">
  {% assign mindful_posts = site.posts | where_exp: "post", "post.categories contains 'mindfulness'" | sort: "date" | reverse %}
  {% for post in mindful_posts %}
    <a class="mindfulness-card" href="{{ post.url }}">
      {% if post.image %}
      <img src="{{ post.image }}" alt="{{ post.title | escape }}" loading="lazy">
      {% endif %}
      <span class="mindfulness-card-title">{{ post.title }}</span>
      <small class="mindfulness-card-date">{{ post.date | date: "%Y-%m-%d" }}</small>
    </a>
  {% endfor %}
</div>

<style>
.mindfulness-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 1.2rem;
  margin: 1.5rem 0;
}
.mindfulness-card {
  display: flex;
  flex-direction: column;
  text-decoration: none;
  border: 1px solid rgba(0,0,0,.12);
  border-radius: 8px;
  overflow: hidden;
  background: rgba(255,255,255,.6);
}
.mindfulness-card img {
  width: 100%;
  aspect-ratio: 3 / 2;
  object-fit: cover;
  display: block;
  margin: 0;
}
.mindfulness-card-title {
  padding: .6rem .8rem .2rem;
  font-weight: 600;
  line-height: 1.3;
}
.mindfulness-card-date {
  padding: 0 .8rem .6rem;
}
@media (prefers-color-scheme: dark) {
  .mindfulness-card {
    border-color: rgba(255,255,255,.18);
    background: rgba(255,255,255,.06);
  }
}
</style>
