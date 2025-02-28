function ue(){import.meta.url,import("_").catch(()=>1),async function*(){}().next()}const R=Symbol.for("RemoteUi::Retain"),L=Symbol.for("RemoteUi::Release"),P=Symbol.for("RemoteUi::RetainedBy");class q{constructor(){this.memoryManaged=new Set}add(t){this.memoryManaged.add(t),t[P].add(this),t[R]()}release(){for(const t of this.memoryManaged)t[P].delete(this),t[L]();this.memoryManaged.clear()}}function U(d){return!!(d&&d[R]&&d[L])}function k(d){if(d==null||typeof d!="object")return!1;const t=Object.getPrototypeOf(d);return t==null||t===Object.prototype}const O="_@f";function G(d){const t=new Map,s=new Map,n=new Map;return{encode:p,decode:i,async call(e,o){const l=new q,c=s.get(e);if(c==null)throw new Error("You attempted to call a function that was already released.");try{const u=U(c)?[l,...c[P]]:[l];return await c(...i(o,u))}finally{l.release()}},release(e){const o=s.get(e);o&&(s.delete(e),t.delete(o))},terminate(){t.clear(),s.clear(),n.clear()}};function p(e,o=new Map){if(e==null)return[e];const l=o.get(e);if(l)return l;if(typeof e=="object"){if(Array.isArray(e)){o.set(e,[void 0]);const u=[],y=[e.map(w=>{const[h,r=[]]=p(w,o);return u.push(...r),h}),u];return o.set(e,y),y}if(k(e)){o.set(e,[void 0]);const u=[],y=[Object.keys(e).reduce((w,h)=>{const[r,a=[]]=p(e[h],o);return u.push(...a),{...w,[h]:r}},{}),u];return o.set(e,y),y}}if(typeof e=="function"){if(t.has(e)){const y=t.get(e),w=[{[O]:y}];return o.set(e,w),w}const u=d.uuid();t.set(e,u),s.set(u,e);const m=[{[O]:u}];return o.set(e,m),m}const c=[e];return o.set(e,c),c}function i(e,o){if(typeof e=="object"){if(e==null)return e;if(Array.isArray(e))return e.map(l=>i(l,o));if(O in e){const l=e[O];if(n.has(l))return n.get(l);let c=0,u=!1;const m=()=>{c-=1,c===0&&(u=!0,n.delete(l),d.release(l))},y=()=>{c+=1},w=new Set(o),h=(...r)=>{if(u)throw new Error("You attempted to call a function that was already released.");if(!n.has(l))throw new Error("You attempted to call a function that was already revoked.");return d.call(l,r)};Object.defineProperties(h,{[L]:{value:m,writable:!1},[R]:{value:y,writable:!1},[P]:{value:w,writable:!1}});for(const r of w)r.add(h);return n.set(l,h),h}if(k(e))return Object.keys(e).reduce((l,c)=>({...l,[c]:i(e[c],o)}),{})}return e}}const v=0,C=1,j=2,_=3,M=5,x=6;function V(d,{uuid:t=X,createEncoder:s=G,callable:n}={}){let p=!1,i=d;const e=new Map,o=new Map,l=Y(y,n),c=s({uuid:t,release(r){u(_,[r])},call(r,a,f){const E=t(),S=w(E,f),[g,b]=c.encode(a);return u(M,[E,r,g],b),S}});return i.addEventListener("message",m),{call:l,replace(r){const a=i;i=r,a.removeEventListener("message",m),r.addEventListener("message",m)},expose(r){for(const a of Object.keys(r)){const f=r[a];typeof f=="function"?e.set(a,f):e.delete(a)}},callable(...r){if(n!=null)for(const a of r)Object.defineProperty(l,a,{value:y(a),writable:!1,configurable:!0,enumerable:!0})},terminate(){u(j,void 0),h(),i.terminate&&i.terminate()}};function u(r,a,f){p||i.postMessage(a?[r,a]:[r],f)}async function m(r){const{data:a}=r;if(!(a==null||!Array.isArray(a)))switch(a[0]){case j:{h();break}case v:{const f=new q,[E,S,g]=a[1],b=e.get(S);try{if(b==null)throw new Error(`No '${S}' method is exposed on this endpoint`);const[A,$]=c.encode(await b(...c.decode(g,[f])));u(C,[E,void 0,A],$)}catch(A){const{name:$,message:H,stack:J}=A;throw u(C,[E,{name:$,message:H,stack:J}]),A}finally{f.release()}break}case C:{const[f]=a[1];o.get(f)(...a[1]),o.delete(f);break}case _:{const[f]=a[1];c.release(f);break}case x:{const[f]=a[1];o.get(f)(...a[1]),o.delete(f);break}case M:{const[f,E,S]=a[1];try{const g=await c.call(E,S),[b,A]=c.encode(g);u(x,[f,void 0,b],A)}catch(g){const{name:b,message:A,stack:$}=g;throw u(x,[f,{name:b,message:A,stack:$}]),g}break}}}function y(r){return(...a)=>{if(p)return Promise.reject(new Error("You attempted to call a function on a terminated web worker."));if(typeof r!="string"&&typeof r!="number")return Promise.reject(new Error(`Can’t call a symbol method on a remote endpoint: ${r.toString()}`));const f=t(),E=w(f),[S,g]=c.encode(a);return u(v,[f,r,S],g),E}}function w(r,a){return new Promise((f,E)=>{o.set(r,(S,g,b)=>{if(g==null)f(b&&c.decode(b,a));else{const A=new Error;Object.assign(A,g),E(A)}})})}function h(){var r;p=!0,e.clear(),o.clear(),(r=c.terminate)===null||r===void 0||r.call(c),i.removeEventListener("message",m)}}function X(){return`${I()}-${I()}-${I()}-${I()}`}function I(){return Math.floor(Math.random()*Number.MAX_SAFE_INTEGER).toString(16)}function Y(d,t){let s;if(t==null){if(typeof Proxy!="function")throw new Error("You must pass an array of callable methods in environments without Proxies.");const n=new Map;s=new Proxy({},{get(p,i){if(n.has(i))return n.get(i);const e=d(i);return n.set(i,e),e}})}else{s={};for(const n of t)Object.defineProperty(s,n,{value:d(n),writable:!1,configurable:!0,enumerable:!0})}return s}function F(){return typeof performance>"u"?Date.now():performance.now()}const W=`
query validation($address: AddressInput!, $locale: String!, $matchingStrategy: MatchingStrategy) {
  validation(address: $address, locale: $locale, matchingStrategy: $matchingStrategy) {
    validationScope
    locale
    fields {
      name
      value
    }
    concerns {
      fieldNames
      code
      type
      typeLevel
      suggestionIds
      message
    }
    id
    suggestions {
      id
      address1
      streetName
      streetNumber
      address2
      line2
      neighborhood
      city
      zip
      provinceCode
      province
      countryCode
    }
  }
}
`;class B extends Error{constructor(t,s){super(`GraphQL fetch failed with network failure or prevented the request from completing: ${s.message}. headers: ${JSON.stringify(Object.fromEntries(t.entries()))}`),this.headers=t,this.error=s}name="AddressValidatorGraphQLFetchNetworkError"}class z extends Error{constructor(t,s){super(`GraphQL fetch failed with status: ${t}, response: ${s}`),this.status=t,this.response=s}name="AddressValidatorHttpError"}const K="validation",Z={"Content-Type":"application/json",Accept:"application/json"};class ee{validationEndpoint;constructor({validationEndpoint:t}){this.validationEndpoint=t}async validate({address:t,locale:s,matchingStrategy:n},p={}){const i=new Headers({...Z,...p.overrideHeaders}),e=await fetch(this.validationEndpoint,{method:"POST",headers:i,body:JSON.stringify({query:W,operationName:K,variables:{address:t,locale:s,matchingStrategy:n}})}).catch(c=>c);if(e instanceof Error)throw new B(i,e);if(!e.ok)throw new z(e.status,await e.text());const{data:o,errors:l}=await e.json();return{data:o.validation,errors:l}}}const te="atlas.shopifysvc.com",N=`https://${te}/graphql`,D="elasticsearch",Q="google",re=new Map([["ELASTICSEARCH_AUTOCOMPLETE",D],["GOOGLE_PLACE_AUTOCOMPLETE",Q]]);class T extends Error{name="AutocompletePredictionError";groupingHash;constructor(t,s){super(t),this.groupingHash=s}}const ne=`
  query predictions($query: String, $countryCode: AutocompleteSupportedCountry!, $locale: String!, $location: LocationInput, $sessionToken: String!, $adapterOverride: String) {
    predictions(query: $query, countryCode: $countryCode, locale: $locale, location: $location, sessionToken: $sessionToken, adapterOverride: $adapterOverride) {
      addressId
      description
      completionService
      matchedSubstrings {
        length
        offset
      }
    }
  }
`,oe=`
  query address($addressId: String!, $locale: String!, $sessionToken: String!, $adapterOverride: String, $extendedFields: Boolean = false) {
    address(id: $addressId, locale: $locale, sessionToken: $sessionToken, adapterOverride: $adapterOverride, extendedFields: $extendedFields) {
      address1
      address2
      city
      country
      countryCode
      province
      provinceCode
      zip
      latitude
      longitude
    }
  }
`,se=`
  query countries($locale: SupportedLocale!) {
    countries(locale: $locale) {
      name
      code
      phoneNumberPrefix
    }
  }
`,ae=async(d,t)=>{const s={addressId:d,locale:t.locale,sessionToken:t.requestToken,adapterOverride:re.get(t.completionService),extendedFields:t.extendedFields},n=await fetch(N,{method:"POST",headers:{"Content-Type":"application/json","X-Shop-Id":t.shopId,"X-Client-Request-Id":t.sourceId},body:JSON.stringify({query:oe,variables:s})});if(!n.ok)throw new T(`Failed to fetch address: ${n.status} ${n.statusText}`,"AutocompletePredictionError::NoDataReturned::AddressQuery");let p;try{p=await n.json()}catch(r){const a=await n.clone().text();throw new T(`Invalid JSON response: ${r}, response: ${a.substring(0,25)}...`,"AutocompletePredictionError::InvalidJson::AddressQuery")}const{data:{address:i}}=p,{address1:e,address2:o,city:l,countryCode:c,provinceCode:u,zip:m,latitude:y,longitude:w}=i,h={latitude:y,longitude:w};return{postalCode:m,city:l,address1:e,address2:o,countryCode:c,zoneCode:u,coordinates:h.latitude&&h.longitude?h:void 0}},ie=async(d,t,s)=>{const n={query:d,countryCode:t.countryCode,location:t.location,locale:t.locale,sessionToken:t.requestToken};[D,Q].includes(String(s))&&(n.adapterOverride=s);const p=F(),i=await fetch(N,{method:"POST",headers:{"Content-Type":"application/json","X-Shop-Id":t.shopId,"X-Client-Request-Id":t.sourceId},body:JSON.stringify({query:ne,variables:n})}),e=F();if(!i.ok)throw new T(`Failed to fetch predictions: ${i.status} ${i.statusText}`,"AutocompletePredictionError::NoDataReturned::AutocompleteQuery");let o;try{o=await i.json()}catch(c){const u=await i.clone().text();throw new T(`Invalid JSON response: ${c}, response: ${u.substring(0,25)}...`,"AutocompletePredictionError::InvalidJson::AutocompleteQuery")}const{data:l}=o;if(!l)throw new T(`No data returned from autocomplete query ${JSON.stringify(o)}`,"AutocompletePredictionError::NoDataReturned::AutocompleteQuery");return{data:l.predictions,duration:{start:p,end:e}}},ce=new ee({validationEndpoint:N}),de=async(d,t,s,n)=>{const p={"X-Shop-Id":n?.shopId||"","X-Client-Request-Id":n?.sourceId||""};return ce.validate({address:d,locale:t,matchingStrategy:s},{overrideHeaders:p})},le=async(d,t)=>{async function s(e){return fetch(N,{method:"POST",headers:{"Content-Type":"application/json","X-Shop-Id":t?.shopId??"","X-Client-Request-Id":t?.sourceId??""},body:JSON.stringify({query:se,variables:{locale:e.replace(/-/,"_").toUpperCase()}})})}const n=await s(d);if(!n.ok)throw new T(`Failed to fetch countries: ${n.status} ${n.statusText}`,"AutocompletePredictionError::NoDataReturned::FetchCountriesWithPhoneNumberPrefix");let p;try{p=await n.json()}catch(e){const o=await n.clone().text();throw new T(`Invalid JSON response: ${e}, response: ${o.substring(0,25)}...`,"AutocompletePredictionError::InvalidJson::FetchCountriesWithPhoneNumberPrefix")}const{data:i}=p;if(!i)throw new T(`No data returned from fetch countries with phone number prefix query ${JSON.stringify(p)}`,"AutocompletePredictionError::NoDataReturned::FetchCountriesWithPhoneNumberPrefix");return i.countries};typeof window<"u"&&V({addEventListener:window.addEventListener.bind(window),removeEventListener:window.removeEventListener.bind(window),postMessage(t,s){window.parent.postMessage(t,"*",s)}},{callable:["search","fetchAddress","validation","fetchCountriesWithPhoneNumberPrefix"]}).expose({search:ie,fetchAddress:ae,validation:de,fetchCountriesWithPhoneNumberPrefix:le});export{ue as __vite_legacy_guard};
