# Troubleshooting

If you have trouble sending LCM messages from the GPS server to the Duckiebots, add a static route for the LCM broadcast URL.

```
sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev <duckietown_eth>
```
