
# Pets, Cattle and the Chaos Monkey – why your pets need to stay away from the datacenter
![cows in a field]({{ site.url }}/assets/cows.png)

Pets, Cattle and the Chaos Monkey – why your pets need to stay away from the datacenter

Take any of your useful software and systems and think about the hardware and IT infrastructure used to run them.First, ask yourself if you know any servers by name, if there are special groups that have specific software components (database servers, middle tier servers/app servers, message brokers etc.) or other special hardware needs.

Second, ask yourself if any one of these has some issue; do you need it repaired?
If you answer yes to either, then you have found your pets, you know them by name, look after their needs and if they are sick you take them to the vet.
On the other hand, some of the hardware has no name, it is generic – if it fails or dies, you just replace it with more generic boxes; these are the cattle. You really don’t care if they go wrong, and if you have demand, you simply increase the herd with more nameless members.
Modern architecture (public or private clouds) demands cattle; it does not want to care about an individual server’s reliability. If a machine fails or becomes overloaded, another generic server can be added to the farm. It is thanks to the cattle machines that you get resilience and the ability to scale on demand.

That said, you need to work with your system and software so they work well with cattle, with no hidden pets that demand special attention, this can be a tough challenge, and that’s where the Chaos Monkey comes in.

Despite its name, the Chaos Monkey is your friend. You turn him lose in your test environments and he constantly jumps around randomly switching machines off in a most ungraceful – but necessary – way.
If you only have cattle in the data center, then all is well. Other boxes simply come on stream to replace the ones that the monkey has stopped. But, he is also designed to sniff out any hidden pets. If the Chaos Monkey switches off a pet, the system goes down until the pet can get its special attention. This way you know you have problems – give that monkey a banana, for he has done his job well.

So the moral of our story, if you want to scale and have high resilience find your pets and take them home, love your cattle and get the Chaos Monkey to make sure all is well.