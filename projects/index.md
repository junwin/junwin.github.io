---
layout: default
title: Projects
permalink: /projects/
---

<h1>Projects</h1>

<p>Project pages and longer engagements. This list shows posts whose categories or tags include "project" or "projects".</p>

{% assign projects_category = site.posts | where_exp: "p", "p.categories contains 'projects'" %}
{% assign project_category = site.posts | where_exp: "p", "p.categories contains 'project'" %}
{% assign projects_tag = site.posts | where_exp: "p", "p.tags contains 'projects'" %}
{% assign project_tag = site.posts | where_exp: "p", "p.tags contains 'project'" %}
{% assign projects = projects_category | concat: project_category | concat: projects_tag | concat: project_tag | uniq %}

{% if projects.size > 0 %}
  <ul class="list-compact">
    {% for post in projects %}
      <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <small class="small">({{ post.date | date: "%Y-%m-%d" }})</small></li>
    {% endfor %}
  </ul>
{% else %}
  <p>No project-tagged posts found. You can navigate to the <a href="{{ '/photography/' | relative_url }}">Photography</a> or <a href="{{ '/writing/' | relative_url }}">Writing</a> pages.</p>
{% endif %}
