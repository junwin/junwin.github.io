---
layout: post
title:  "Continuous Delivery"
date:   2017-01-24 15:45:18 -0600
categories: jekyll update
---

Continuous Integration
Continuous Delivery


**Continuous Integration(CI)**

* Enable you to make changes quickly and easily
* Auto Detect code committed
* Build
* Run Unit Tests
* Deploy Service
* Run Service tests

**Jez Humble - 3 questions**

* Do you check into the mainline each day? This includes integrating short lived branches to the main.
* Do you have a suite of tests to validate your changes?
* When the build is broken is it the #1 team priority to fix it?


**Options**

* Global Build for all services (clumsy when there are numerous service, plus how do you know what to deploy)
* Single Source tree N builds (at a folder level) - can encourage coupling by checking in changes for N services once
* Single CI Process per Service - each service has its own repo(code and tests) and build process - implies a repo per service and a build for each service

Principle
* Avoid lock-step releases

**Exceptions**
At the start may well be better to go for option 1 while the service boundaries are in some flux, then move them to 3 as API's stabilize

**Continuous Delivery**

* Build Pipelines and continuous delivery(compile and unit tests, Slow Tests, deploy UAT, performance tests, deploy to prod)
* Model the standard release process
* Aim is to get quick feedback on the production readiness of each and every check-in
* One pipeline per service

**Artifacts**
* jars, wars or rpm, NuGet, chocolaty
* What about multiple OS's - Custom VM images
* Netflicks bake their services as AWS images (AMI's Amazon machine images)

**Immutable Server**
To avoid configuration drift, you need the pipeline to build an entire machine.

**Service Configuration**
Avoid building and artifact per environment (how can you really validate, plus need to store secure ids etc.), use settings file or configuration system.

**Mapping Services to machines(Hosts)**
Think of hosts as an OS where a service can be deployed - physical hardware can support 1..N hosts

N Services per host is simple but has issues (performance of one affects others, different dependencies, risk, ripple affect, may results in centralized team to manage host, limits deployment options)

Application Containers (IIS, Tomcat) Provides clustering, monitoring and more efficient use of resources, but you need to buy into the technology stack, the features offered may constrain and not scale, lifecycle management is more complex, monitoring and analysis is harder if N services share some container

Single Service per host - can reduce single points of failure, simplify monitoring and remediation, an outage only affects a single host, can scale service independently of each other, can use services deployed as images and the immutable server pattern.
Note that single service per host can be quite expensive an extreme would be where a host is a physical machine, hence VMs and containers like Docker

Platform as a Service PaaS - takes an artifact like a war file or gem and provisions and runs them.

**Automation**
A must if you go down a Service per Host route

**Appendix - Artifacts**
You add web components to a J2EE application in a package called a web application archive (WAR), which is a JAR similar to the package used for Java class libraries. A WAR usually contains other resources besides web components, including:

Server-side utility classes (database beans, shopping carts, and so on).
Static web resources (HTML, image, and sound files, and so on)
Client-side classes (applets and utility classes)
A WAR has a specific hierarchical directory structure. The top-level directory of a WAR is the document root of the application. The document root is where JSP pages, client-side classes and archives, and static web resources are stored.

So a .war is a .jar, but it contains web application components and is laid out according to a specific structure. A .war is designed to be deployed to a web application server such as Tomcat or Jetty or a Java EE server such as JBoss or Glassfish.
