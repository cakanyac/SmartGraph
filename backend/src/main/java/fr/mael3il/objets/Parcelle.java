package fr.mael3il.objets;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.neo4j.driver.types.Node;

public class Parcelle {
    String id;
    String nom;
    Double superficie;
    String culture;
    @JsonProperty("type_sol")
    String typeSol;
    Double latitude;
    Double longitude;

    public Parcelle(String id, String nom, Double superficie, String culture, String typeSol, Double latitude, Double longitude) {
        this.id = id;
        this.nom = nom;
        this.superficie = superficie;
        this.culture = culture;
        this.typeSol = typeSol;
        this.latitude = latitude;
        this.longitude = longitude;
    }

    public String getId() {
        return id;
    }

    public String getNom() {
        return nom;
    }

    public Double getSuperficie() {
        return superficie;
    }

    public String getCulture() {
        return culture;
    }

    public String getTypeSol() {
        return typeSol;
    }

    public Double getLatitude() {
        return latitude;
    }

    public Double getLongitude() {
        return longitude;
    }

    public static Parcelle from(Node node) {
        return new Parcelle(
                node.get("id").asString(),
                node.get("nom").asString(),
                node.get("superficie").asDouble(),
                node.get("culture").asString(),
                node.get("type_sol").asString(),
                node.get("latitude").asDouble(),
                node.get("longitude").asDouble()
        );
    }
}