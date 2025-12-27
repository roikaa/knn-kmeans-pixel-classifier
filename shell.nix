{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
    packages = [
      (pkgs.python312.withPackages(p: with p; [
        numpy
        pillow
        matplotlib
        scikit-learn
      ]))

    ];
}
