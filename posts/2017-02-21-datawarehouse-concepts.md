---
layout: post
title:  "Datawarehouse concepts"
date:   2017-02-21 15:45:18 -0600
categories: jekyll update
---


I have been revisiting strategies of using data warehouses in conjunction with transactional systems so decided to put down some of the concepts here.

A data warehouse aggregates information from different often heterogeneous sources. In many cases, an integration layer and ETL is used to build the data warehouse.

Frequently, a set of data marts is constructed using ETL to pull data from the central warehouse. These data marts focus on a single area of the business. Hence while distinct domains may exist in the data warehouses, information from one or more of the can be pulled into a data mart.

Facts are quantitative data about some aspect of a business or domain. Some facts relate to transactions e.g. sales; others might be snapshots of data at some point in time, others can be aggregations.

**Dimensional storage of data.** In this case, the data is frequently modeled as a star schema (see Ralph Kimball) In this model transactional data is modeled as facts and any information that provides context into dimensions. For example, a stock trade might have facts (quantity, price) and dimensions (account name, product, trade date, period, location) the dimensions are *not* normalized.

**Normalized storage of data.**
In this model, tables use normalized data (3NF) to store data. Business domains(accounts, trades, products, etc.) or processes are used to group the tables. 

Dimensional models can be simpler to use, but harder to change and adapt, whereas normalized models can be easier to extend and change but present a steeper learning curve.

**Strategies for designing a data warehouse** Bottom up - data marts are designed to meet specific business needs, and these marts are then analyzed to form the facts and dimensions in the enterprise model. Top down - the schema is decomposed from an enterprise model.

**References**
Start scheme https://en.wikipedia.org/wiki/Star_schema  
Snowflake schema https://en.wikipedia.org/wiki/Snowflake_schema