import * as THREE from 'three';

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000);

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Takes 0 -> face, 1 -> vertex, 2 -> edge.
let currentMode = 0;

document.addEventListener('keydown', (event) => {
    if (event.key == 'q') {
        isRotatingNegativeZ = true;
    }
    if (event.key == 'w') {
        isRotatingPositiveZ = true;
    }
    if (event.key == 'e') {
        isRotatingNegativeY = true;
    } 
    if (event.key == 'r') {
        isRotatingPositiveY = true;
    } 
    if (event.key == 't') {
        isRotatingNegativeX = true;
    }
    if (event.key == 'y') {
        isRotatingPositiveX = true;
    }
    if (event.key == 'v') {
        currentMode = (currentMode + 1) % 3
    }
})


document.addEventListener('keyup', (event) => {
    if (event.key == 'q') {
        isRotatingNegativeZ = false;
    }
    if (event.key == 'w') {
        isRotatingPositiveZ = false;
    }
    if (event.key == 'e') {
        isRotatingNegativeY = false;
    } 
    if (event.key == 'r') {
        isRotatingPositiveY = false;
    } 
    if (event.key == 't') {
        isRotatingNegativeX = false;
    }
    if (event.key == 'y') {
        isRotatingPositiveX = false;
    }
})


const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const parent = new THREE.Object3D();
scene.add(parent);

camera.position.z = 5;

// Can modify later if necessary for more degrees of freedom.
const rotationSpeed = 0.1;
let isRotatingNegativeZ = false;
let isRotatingPositiveZ = false;
let isRotatingNegativeX = false;
let isRotatingPositiveX = false;
let isRotatingNegativeY = false;
let isRotatingPositiveY = false;

let rendered;
let previousMode;

function animate() {
    if (currentMode !== previousMode) {
        if (rendered) {
            parent.remove(rendered);
            if (rendered.geometry) rendered.geometry.dispose();
            if (rendered.material) rendered.material.dispose();
        }

        switch (currentMode) {
            case 0:
                const basicMaterial = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
                rendered = new THREE.Mesh( geometry, basicMaterial );
                break;

            case 1:
                const pointsMaterial = new THREE.PointsMaterial({
                    color: 0x0ff00,
                    size: 0.05,
                    sizeAttenuation: true
                });
                rendered = new THREE.Points(geometry, pointsMaterial);
                break;

            case 2:
                const edges = new THREE.EdgesGeometry( geometry ); 
                rendered = new THREE.LineSegments(edges, new THREE.LineBasicMaterial( { color: 0xffffff } ) ); 
                break;

            default:
                break;
        }

        parent.add( rendered );
        previousMode = currentMode;
    }

    if (isRotatingNegativeX) {
        parent.rotation.x -= rotationSpeed;
    }
    if (isRotatingPositiveX) {
        parent.rotation.x += rotationSpeed;
    }
    if (isRotatingNegativeY) {
        parent.rotation.y -= rotationSpeed;
    }
    if (isRotatingPositiveY) {
        parent.rotation.y += rotationSpeed;
    }
    if (isRotatingNegativeZ) {
        parent.rotation.z -= rotationSpeed;
    }
    if (isRotatingPositiveZ) {
        parent.rotation.z += rotationSpeed;
    }
    
	renderer.render( scene, camera );
}

renderer.setAnimationLoop( animate );


