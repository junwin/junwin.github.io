---
layout: default
title: John Unwin
---

<section class="hero">
  <div class="container">
    <div class="intro">
      <h1>{{ site.title }}</h1>
      <p class="small">{{ site.description }}</p>
      <p>I write about development, systems, creativity, and the occasional photograph. Use the links above to get to Photography, Projects, or Writing.</p>
    </div>
  </div>
</section>

<section>
  <div class="container">
    <h2>Photography</h2>
    <div id="photo-grid" class="photo-grid" data-json="{{ '/gallery/gallery.json' | relative_url }}" data-limit="8"></div>
    <p class="small"><a href="{{ '/gallery/' | relative_url }}">All photos →</a></p>
    <link rel="stylesheet" href="{{ '/assets/css/gallery.css' | relative_url }}">
    <script src="{{ '/assets/js/gallery.js' | relative_url }}" defer></script>
  </div>
</section>

<section>
  <div class="container">
    <h2>Latest writing</h2>
    <ul class="list-compact">
      {% for post in site.posts limit:6 %}
        <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small class="small">({{ post.date | date: "%Y-%m-%d" }})</small></li>
      {% endfor %}
    </ul>
    <p class="small"><a href="{{ '/writing/' | relative_url }}">All writing →</a></p>
  </div>
</section>
