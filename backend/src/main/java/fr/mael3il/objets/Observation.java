package fr.mael3il.objets;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.neo4j.driver.types.Node;

import java.time.ZonedDateTime;

public class Observation {
    String id;

    @JsonProperty("result_time")
    ZonedDateTime resultTime;

    String unit;

    @JsonProperty("observation_type")
    String observationType;

    @JsonProperty("made_by_sensor")
    String madeBySensor;

    @JsonProperty("numeric_value")
    Double numericValue;

    public Observation(String id,
                       ZonedDateTime resultTime,
                       String unit,
                       String observationType,
                       String madeBySensor,
                       Double numericValue) {
        this.id = id;
        this.resultTime = resultTime;
        this.unit = unit;
        this.observationType = observationType;
        this.madeBySensor = madeBySensor;
        this.numericValue = numericValue;
    }

    public String getId() {
        return id;
    }

    public ZonedDateTime getResultTime() {
        return resultTime;
    }

    public String getUnit() {
        return unit;
    }

    public String getObservationType() {
        return observationType;
    }

    public String getMadeBySensor() {
        return madeBySensor;
    }

    public Double getNumericValue() {
        return numericValue;
    }

    public static Observation from(Node node) {
        return new Observation(
                node.get("id").asString(),
                node.get("resultTime").asZonedDateTime(),
                node.get("unit").asString(),
                node.get("observationType").asString(),
                node.get("madeBySensor").asString(),
                node.get("numericValue").asDouble()
        );
    }
}
