{
    description = "Blucheck";

    inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    outputs = { self, nixpkgs }:
        let
            system = "x86_64-linux";
            pkgs = import nixpkgs {
                inherit system;
                config.allowUnfree = true;
            };
        in {
            devShells.${system}.default = pkgs.mkShell {
                packages = with pkgs; [
                    (python313.withPackages (ps: with ps; [
                        transformers
                        datasets
                        torch
                        pandas
                        scikit-learn
                        evaluate
                        gradio
                        accelerate
                    ]))
                ];

                shellHook = ''
                    echo "Python dev environment ready: $(python --version)"
                '';
            };
        };
}
