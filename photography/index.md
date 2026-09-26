---
layout: default
title: Photography
permalink: /photography/
---

<h1>Photography</h1>

<p>Photographs and visual work. The page below lists posts that include an image in their front matter.</p>

{% assign with_images = site.posts | where_exp: "p", "p.image" %}
{% if with_images.size > 0 %}
  <div class="post-grid">
    {% for post in with_images %}
      <article class="card">
        <a href="{{ post.url | relative_url }}">
          <img src="{{ post.image | relative_url }}" alt="{{ post.title }}">
          <h3>{{ post.title }}</h3>
        </a>
      </article>
    {% endfor %}
  </div>
{% else %}
  <p>No photographic posts found.</p>
{% endif %}
