const cacheName = 'v1';

const cacheAssets = [
'index.html',
'predict.js'
];

//install
self.addEventListener('install', (e)=>{
	console.log("Service Worker: Installed");


});


self.addEventListener('activate', (e)=>{
	console.log("Service Worker: Activated");
	// remove caches
	e.waitUntil(
		caches.keys().then(cacheNames =>{
			return Promise.all(
				cacheNames.map(cache => {
					if(cache !== cacheName){
						console.log('Service Worker: Clear cache');
						return caches.delete(cache);
					}
				}))
		})
		);
});

// Call Fetch Event

self.addEventListener('fetch', e =>{
	console.log('Service Worker: Fetching');

	e.respondWith(
		fetch(e.request)
			.then(res => {

				const resClone = res.clone();

				caches.open(cacheName).then(cache =>{

					cache.put(e.request, resClone);
				});
				return res;
		}).catch(err => caches.match(e.request).then(res=>res))
);

});
