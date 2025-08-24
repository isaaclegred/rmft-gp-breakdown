The workflow proceeds as follows.

```
./condition # construct processes
./draw      # draw samples, solve TOV equations
./plot      # plot draws
./analyze   # analyse micro and macro relations, compute quantiles
```

The individual priors (conditiond on separate compositions) can be combined in to a single marginalized prior.
This will create composition-marginalized priors for each maximum-pressure as well as a prior marginalized over the maxium-pressure.

```
./combine
```
