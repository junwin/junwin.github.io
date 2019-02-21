---
layout: post
title:  "SOA, ESB and event driven architectures -  oh my!!"
date:   2017-01-25 08:05:18 -0600
categories: jekyll update
---

**Service Orientated Architecture (SOA)**
SOA is an architectural concept where different services can be discovered and their interfaces are developed so that different systems can use them irrespective of the development platform. Its key goal is to facilitate reuse of common services with an organization or some wider ecosystem. The main features are:
* Services are can be used without consideration of how they were developed or where they are deployed.
* The employ a request response model, where communication is initiated by the client/consumer, and results are synchronously returned from the service.
* Only one client/consumer can use the service at a time.

A draw back to the SOA model is that it is centralized and additional methods are needed to provide failover and scaling.
Clients are coupled to the service by consuming its interface, thus changes to the service can impact the client.


**Enterprise Service Bus bases architecture (ESB)** An ESB usually has a centralized bus that is used to route messages to different components. The ESB provides additional features to facilitate translating and converting the format of the requests and responses that concern them as well as filtering and content-based routing capabilities. Despite the features provided by an ESB, the essential architecture is still point to point SOA and has the same problems with coupling and centralization.

**Event Driven Architecture (EDA)** An EDA uses messaging fabric to connect a set of applications. It does not use a point to point design, any given application can usually see all the events raised by other applications. The EDA provides the equivalent filtering, transforming and routing capabilities as an ESB. However, the EDA puts emphasis on decoupling and decentralization. An EDA generally uses a canonical model to decouple different components, in trading applications this is often based on FIX protocol. 

**Staged Event Driven Architecture (SEDA)** (SEDA)  decomposes a complex system into a series of stages connected by queues. SEDA is a variant of an EDA that explicitly exposes a pipe-line of operations, the main benefits are:
* It can produce a highly scalable solution, with no centralized points of failure.
* It simplifies the rollout of additional components, and eases the on-going maintenance overheads. 

It is suited to circumstances where actions depend on some event, for example, when some data becomes available (an event) a pipe line of actions are performed. In most cases these events are modeled as messages that can be sent to one or more components (stages).
Events flow from producers to components that consume (and may produce) along a predefined sequence with no use of a central routing service. 

