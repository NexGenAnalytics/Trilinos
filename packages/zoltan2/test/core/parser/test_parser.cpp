#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#if HAVE_YAML
#include "yaml-cpp/yaml.h"
#endif // HAVE_YAML


void parse_mini_em_sample(std::string filename) {

    // ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // ------------------------------------------------- Trilinos/packages/panzer/mini-em/example/BlockPrec/main.cpp (line 195) ---------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Using yaml-cpp                                                                // Using Teuchos::ParameterList

    YAML::Node input_params = YAML::LoadFile(filename);                              // Teuchos::RCP<Teuchos::ParameterList> input_params = Teuchos::rcp(new Teuchos::ParameterList("User_App Parameters"));
    YAML::Node physicsBlock_pl = input_params["Physics Blocks"];                     // Teuchos::ParameterList &physicsBlock_pl = input_params->sublist("Physics Blocks");

    YAML::Node physicsEqSet = physicsBlock_pl["Maxwell Physics"]["Maxwell Physics"]; // Teuchos::ParameterList& physicsEqSet = physicsBlock_pl.sublist("Maxwell Physics").sublist("Maxwell Physics");

    const std::string physicsTypeStr = physicsEqSet["Type"].as<std::string>();       // const std::string physicsTypeStr = physicsEqSet.get<std::string>("Type");

    std::string physics;                                                             // physicsType physics; (for demonstration purposes, we replace physicsType with std::string);

    if (physicsTypeStr == "Maxwell") {                                               // if (physicsTypeStr == "Maxwell")
        physics = "MAXWELL";                                                         //    physics = MAXWELL;
    } else if (physicsTypeStr == "Darcy") {                                          // else if (physicsTypeStr == "Darcy")
        physics = "DARCY";                                                           //    physics = DARCY;
    } else {                                                                         // else
        std::cout << "Error: invalid physicsTypeStr" << std::endl;                   //    TEUCHOS_ASSERT(false);
    }

    int basis_order = physicsEqSet["Basis Order"].as<int>();                         // basis_order = physicsEqSet.get("Basis Order", basis_order);

    physicsEqSet["Integration Order"] = 2 * basis_order;                             // physicsEqSet.set("Integration Order", 2*basis_order);

    std::cout << "Physics Type: " << physics << std::endl;
    std::cout << "Basis Order:  " << basis_order << std::endl;
    std::cout << "Integration Order: " << physicsEqSet["Integration Order"].as<int>() << std::endl;
}

void parse_wishlist(std::string filename) {
    YAML::Node input_file = YAML::LoadFile(filename);
    YAML::Node wishlist = input_file["Wishlist"];

    // Check Indentation
    const std::vector<std::string> indentation = wishlist["Indentation"].as<std::vector<std::string>>();
    std::cout << "\nCheck correct indentation:" << std::endl;
    for (const auto& element : indentation) {
        std::cout << element << " ";
    } std::cout << std::endl;

    // Check Wrong indentation
    const std::vector<std::string> wrong_indentation = wishlist["Wrong indentation"].as<std::vector<std::string>>();
    std::cout << "\nCheck incorrect indetation:" << std::endl;
    for (const auto& element : wrong_indentation) {
        std::cout << element << " ";
    } std::cout << std::endl;

    // List of lists
    const std::vector<std::vector<int>> list_of_lists = wishlist["List of lists"].as<std::vector<std::vector<int>>>();
    std::cout << "\nCheck list of lists:" << std::endl;
    for (const auto& list : list_of_lists) {
        for (const auto& elt : list) {
            std::cout << elt << " ";
        } std::cout << std::endl;
    }

    // List of lists of lists
    const std::vector<std::vector<std::vector<int>>> list_of_lists_of_lists = wishlist["List of lists of lists"].as<std::vector<std::vector<std::vector<int>>>>();
    std::cout << "\nCheck list of lists:" << std::endl;
    for (const auto& lists : list_of_lists_of_lists) {
        for (const auto& list : lists) {
            std::cout << "[ ";
            for (const auto& elt : list) {
                std::cout << elt << " ";
            } std::cout << "]";
        } std::cout << std::endl;
    }

    // Line continuation
    const std::vector<std::vector<int>> line_continuation = wishlist["Line continuation"].as<std::vector<std::vector<int>>>();
    std::cout << "\nCheck line continuation:" << std::endl;
    for (const auto& list : line_continuation) {
        for (const auto& elt : list) {
            std::cout << elt << " ";
        } std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {

    std::string executable_path = std::filesystem::path(argv[0]).parent_path().string();
    std::string filename = executable_path + "/config.yaml";

    // std::string filename = "config.yaml" // doesn't work
    std::cout << "Parsing " << filename << "\n" << std::endl;

#if HAVE_YAML
    parse_mini_em_sample(filename);
    parse_wishlist(filename);
#endif

    return 0;
}

