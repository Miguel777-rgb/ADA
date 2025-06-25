// Espera a que todo el contenido de la página se haya cargado
document.addEventListener('DOMContentLoaded', () => {

    // Selecciona todos los botones que tienen el atributo 'data-url'
    const redirectButtons = document.querySelectorAll('button[data-url]');

    // Recorre cada botón encontrado
    redirectButtons.forEach(button => {
        // Añade un "escuchador de eventos" para el clic
        button.addEventListener('click', () => {
            // Obtiene la URL guardada en el atributo 'data-url' del botón
            const url = button.dataset.url;

            // Muestra en la consola la URL a la que se va a redirigir (útil para depurar)
            console.log(`Redirigiendo a: ${url}`);
            
            // Redirige la ventana del navegador a la nueva URL
            window.location.href = url;
        });
    });

});