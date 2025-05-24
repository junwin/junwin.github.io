
# The futile search for the canonical
![cityscape]({{ site.url }}/assets/L1070928FS.png)
In [Microservice Architecture (Irakli Nadareishvili, Matt McLarty, Michael Amundsen, and Ronnie Mitra)](http://amzn.to/2k5lme4) the authors provide an interesting story about finding the ideal.

In a US Air force study detailing pilot error, Lt. Gilbert Daniels attempted to identify what made up an average pilot. Based on some simple dimensions (height, chest size, etc.), and allowing a 30% margin around the numerical average(for all pilots the study) to consider an individual average. The study found that with ten measured dimensions and 4000 subjects not one person was average, the implication was that designing a cockpit for an average pilot would yield a design that fitted no one. 

Todd Rose discusses the problem in his book [The End of Average](http://amzn.to/2k5icap). Rose describes this as the *principle of jaggedness*. The principle of jaggedness applies to software architecture and design, time spent on developing the ideal canonical model or service interface is likely to lead to a service that does not fit well with other systems or services in its environment.

A better approach is to understand the main features of the design and provide a mechanism to accommodate variation and change. In terms of an aircraft cockpit this would be to allow a pilot to adjust the steering column and other main controls, in terms of a canonical model of an object it would be a clean approach to add additional attributes.

A practical example is the use of FIX an industry standard used in electronic trading to define domain objects such as trades, orders, and prices. 
These definitions cover most of the attributes and groups needed and then provide a clear way to extend these as required. For example to add additional price analytics rather than spend much time attempting to come up with some **ideal** canonical model. 

The benefits are:
* A sharp reduction in the time taken to define a useful model.
* A generally good fit to business domains and the ability to adjust a better fit.
* In the case of FIX, the use of a well know protocol reduces learning curves and provides documentation.

In summary, it is not possible to achieve an "ideal" model in non-trivial systems, so don't waste too much time finding one.