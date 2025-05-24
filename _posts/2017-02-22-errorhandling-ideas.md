---
layout: post
title:  "Working out a strategy for error handling."
date:   2017-02-21 15:45:18 -0600
categories: jekyll update
---

Error handling is an important part of any large system and developing a coherent approach can be difficult.

First, there are some questions to ask yourself:

* Will the handling detect errors and issues or will it need to perform a corrective action?
* Will they system actively detect errors, for example, flagging behavior that breaks some threshold or will the detection be passive for example when some component throws and exception?
* When an error happens, who needs to be informed, for example, business or operations?
* What methods are available to raise and propagate error reports in the organization - consider that an existing framework may be in use.
* Do you need to provide a set of instructions/wiki/knowledgebase to resolve all errors raised?
* Are the guidelines for raising and handling exceptions?
* Are the different types of error understood, for example, different system/infrastructure errors versus business or data errors?