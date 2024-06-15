# otel backend

This is a simple otel backend that can receive logs, metrics and traces.
The traces endpoint is expecting cilium traces feeding them into
a gnn model with convolutional layers for the edge features.

## install

```bash
```

## lint/format

```bash
poetry run ruff check . --fix

# or if nix
ruff check . --fix
```

## run app

```bash
poetry run uvicorn otel_backend.app:app --host 0.0.0.0 --port 8000
```

## nix env fix

```nix
{ pkgs ? import <nixpkgs> {}, stdenv ? pkgs.stdenv }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.poetry
  ];

  shellHook = ''
    # fixes libstdc++ issues and libgl.so issues
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
```
