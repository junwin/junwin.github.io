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
    <h2>Featured</h2>
    {% assign featured = site.posts | where_exp: "p", "p.image" %}
    {% if featured.size > 0 %}
      <div class="post-grid">
        {% for post in featured limit:6 %}
          <article class="card">
            <a href="{{ post.url | relative_url }}">
              <img src="{{ post.image | relative_url }}" alt="{{ post.title }}">
              <h3>{{ post.title }}</h3>
            </a>
          </article>
        {% endfor %}
      </div>
      <p class="small"><a href="{{ '/photography/' | relative_url }}">See more photography →</a></p>
    {% else %}
      <p>No featured images found.</p>
    {% endif %}
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
