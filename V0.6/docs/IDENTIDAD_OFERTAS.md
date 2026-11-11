# Auditoría de identidad de ofertas

Se revisaron los 12 CSV originales y sus 2.846.181 registros. No se fusionó ni eliminó ninguna fila.

## Campos que diferencian registros

1. `Codigoitem` identifica la línea de adquisición. `Correlativo` no añadió separación respecto de `Codigoitem`.
2. `CodigoProveedor` y `CodigoSucursalProveedor` distinguen la mayoría de las ofertas por línea.
3. `Nombre de la Oferta` fue el diferenciador adicional más útil: separó 12.871 de los 21.428 grupos inicialmente repetidos.
4. `FechaEnvioOferta`, `Moneda de la Oferta`, `Cantidad Ofertada`, `MontoUnitarioOferta` y `Valor Total Ofertado` separan otros casos.
5. `RutProveedor`, `NombreProveedor` y `RazonSocialProveedor` no separaron grupos adicionales, porque eran constantes dentro de cada proveedor repetido.

## Resultado de la clave

| Nivel | Claves distintas | Grupos repetidos | Filas en grupos repetidos |
|---|---:|---:|---:|
| Línea | 589.669 | 451.511 | 2.708.023 |
| Línea + proveedor + sucursal | 2.789.230 | 21.428 | 57.656 |
| Más nombre de oferta | 2.806.391 | 9.950 | 29.017 |
| Más fecha, moneda, cantidad y montos | 2.209.327 completas | 1.835 | 3.819 |
| Archivo + registro | 2.846.181 | 0 | 0 |

La clave de negocio tiene campos ausentes en algunas filas y todavía puede repetirse. Por esa razón no se utiliza para deduplicar. La identidad definitiva es `_archivo + _registro_archivo`, mientras que la licitación completa se mantiene en una sola partición temporal.
