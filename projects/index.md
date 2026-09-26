---
layout: default
title: Projects
permalink: /projects/
---

<h1>Projects</h1>

<p>Project pages and longer engagements. This list shows posts whose categories or tags include "project" or "projects".</p>

{% assign projects = site.posts | where_exp: "p", "p.categories contains 'projects' or p.categories contains 'project' or p.tags contains 'projects' or p.tags contains 'project'" %}

{% if projects.size > 0 %}
  <ul class="list-compact">
    {% for post in projects %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small class="small">({{ post.date | date: "%Y-%m-%d" }})</small></li>
    {% endfor %}
  </ul>
{% else %}
  <p>No project-tagged posts found. You can navigate to the <a href="{{ '/photography/' | relative_url }}">Photography</a> or <a href="{{ '/writing/' | relative_url }}">Writing</a> pages.</p>
{% endif %}
